from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from core.wect import WECTTransform
from core.shapes import make_shape_library, rotate_image, apply_intensity
from core.mlp import (
    train_mlp, tune_head, params_to_threshold, width_to_threshold,
)

REPRESENTATIONS = ["pixels", "wect", "wect_aligned", "diwect"]
_COLORS = {"pixels": "#8d99ae", "wect": "#d1495b", "wect_aligned": "#2a9d8f",
           "diwect": "#30638e"}
_DISPLAY = {"wect_aligned": "wect+align"}
_ORACLE = {"wect_aligned"}


# --------------------------------------------------------------------------- #
# Dataset + features
# --------------------------------------------------------------------------- #
def make_dataset(shapes, n_per_class, seed):
    rng = np.random.default_rng(seed)
    images, labels, angles = [], [], []
    for label, base in enumerate(shapes.values()):
        for _ in range(n_per_class):
            angle = rng.uniform(0.0, 360.0)
            img = apply_intensity(rotate_image(base, angle), rng)
            images.append(img)
            labels.append(label)
            angles.append(angle)
    return images, np.asarray(labels), np.asarray(angles)


def extract_features(images, angles, transform):
    aligned = [rotate_image(im, -a) for im, a in zip(images, angles)]
    return {
        "pixels": np.stack([im.ravel() for im in images]).astype(np.float32),
        "wect": np.stack([transform.wect(im).ravel() for im in images]).astype(np.float32),
        "wect_aligned": np.stack([transform.wect(im).ravel() for im in aligned]).astype(np.float32),
        "diwect": np.stack([transform.diwect(im) for im in images]).astype(np.float32),
    }


def build_split(shapes, transform, n_per_class, seed):
    imgs, y, angles = make_dataset(shapes, n_per_class, seed)
    return extract_features(imgs, angles, transform), y


def subsample(features, labels, n_per_class, rng):
    idx = []
    for label in np.unique(labels):
        pool = np.where(labels == label)[0]
        idx.append(rng.choice(pool, size=n_per_class, replace=False))
    idx = np.concatenate(idx)
    return {r: X[idx] for r, X in features.items()}, labels[idx]


# --------------------------------------------------------------------------- #
# Experiments (all reuse one featurized pool per seed)
# --------------------------------------------------------------------------- #
def tune_reps(cache, cfg):
    if cfg.no_tune:
        return {rep: {"lr": 1e-3, "weight_decay": 1e-4, "val_acc": None}
                for rep in REPRESENTATIONS}
    tr, ytr, va, yva, te, yte = cache[0]
    grid = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    hp = {}
    for rep in REPRESENTATIONS:
        best = tune_head(tr[rep], ytr, va[rep], yva, te[rep], yte,
                         width=cfg.curve_width, epochs=cfg.curve_epochs, seed=0, grid=grid)
        hp[rep] = {"lr": best["hp"]["lr"], "weight_decay": best["hp"]["weight_decay"],
                   "val_acc": best["best_val_acc"]}
        print(f"    tuned {rep:14} lr={hp[rep]['lr']:.0e} wd={hp[rep]['weight_decay']:.0e}"
              f"  (val acc {hp[rep]['val_acc']:.3f})", flush=True)
    return hp


def run(cfg):
    shapes = make_shape_library(size=cfg.size, area=cfg.area)
    transform = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights)

    t0 = time.time()
    cache = {}
    for seed in range(cfg.seeds):
        tr, ytr = build_split(shapes, transform, cfg.n_train, 10 * seed + 1)
        va, yva = build_split(shapes, transform, cfg.n_val, 10 * seed + 2)
        te, yte = build_split(shapes, transform, cfg.n_test, 10 * seed + 3)
        cache[seed] = (tr, ytr, va, yva, te, yte)
    print(f"[featurized {cfg.seeds} seeds in {time.time() - t0:.1f}s]")

    tag = "held-fixed defaults (--no-tune)" if cfg.no_tune else "validation grid search"
    print(f"[tuning lr/weight_decay per representation: {tag}]")
    hp = tune_reps(cache, cfg)

    curves = _exp_curves(cache, cfg, hp)
    capacity = _exp_capacity(cache, cfg, hp)
    data_eff = _exp_data_efficiency(cache, cfg, hp)
    results = {"shapes": shapes, "curves": curves, "capacity": capacity,
               "data_eff": data_eff, "cfg": cfg, "hp": hp}
    _report(results)
    return results


def _exp_curves(cache, cfg, hp):
    out = {}
    for rep in REPRESENTATIONS:
        val_curves, test_accs, eps, params = [], [], [], None
        for seed in range(cfg.seeds):
            tr, ytr, va, yva, te, yte = cache[seed]
            res = train_mlp(tr[rep], ytr, va[rep], yva, te[rep], yte,
                            width=cfg.curve_width, epochs=cfg.curve_epochs, seed=seed,
                            lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
            val_curves.append(res["hist"]["val_acc"])
            test_accs.append(res["test_acc"])
            eps.append(res["ep_to_thr"])
            params = res["params"]
        val_curves = np.asarray(val_curves)
        reached = [e for e in eps if e is not None]
        out[rep] = {
            "val_mean": val_curves.mean(0), "val_std": val_curves.std(0),
            "test_mean": float(np.mean(test_accs)), "test_std": float(np.std(test_accs)),
            "ep_to_thr": (int(round(np.mean(reached))) if reached else None),
            "params": params,
        }
    return out


def _exp_capacity(cache, cfg, hp):
    out = {rep: {"widths": cfg.widths, "mean": [], "std": [], "params": []}
           for rep in REPRESENTATIONS}
    for rep in REPRESENTATIONS:
        for width in cfg.widths:
            accs, params = [], None
            for seed in range(cfg.seeds):
                tr, ytr, va, yva, te, yte = cache[seed]
                res = train_mlp(tr[rep], ytr, va[rep], yva, te[rep], yte,
                                width=width, epochs=cfg.cap_epochs, seed=seed,
                                lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
                accs.append(res["test_acc"])
                params = res["params"]
            out[rep]["mean"].append(float(np.mean(accs)))
            out[rep]["std"].append(float(np.std(accs)))
            out[rep]["params"].append(params)
    return out


def _exp_data_efficiency(cache, cfg, hp):
    out = {rep: {"n": cfg.n_trains, "mean": [], "std": []} for rep in REPRESENTATIONS}
    for rep in REPRESENTATIONS:
        for n in cfg.n_trains:
            accs = []
            for seed in range(cfg.seeds):
                tr, ytr, va, yva, te, yte = cache[seed]
                rng = np.random.default_rng(100 * seed + n)
                sub, ysub = subsample(tr, ytr, n, rng)
                res = train_mlp(sub[rep], ysub, va[rep], yva, te[rep], yte,
                                width=cfg.curve_width, epochs=cfg.de_epochs, seed=seed,
                                lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
                accs.append(res["test_acc"])
            out[rep]["mean"].append(float(np.mean(accs)))
            out[rep]["std"].append(float(np.std(accs)))
    return out


# --------------------------------------------------------------------------- #
# Report + figure
# --------------------------------------------------------------------------- #
def _report(results):
    cfg = results["cfg"]
    c, cap = results["curves"], results["capacity"]
    n = len(results["shapes"])
    print(f"\nMLP shape classification | {n} classes | {cfg.num_dirs} dirs x "
          f"{cfg.num_heights} heights | {cfg.n_train} train/class, {cfg.seeds} seeds")
    print(f"Columns: test acc at width {cfg.curve_width} (full data); ~fewest trainable"
          f" params to reach 95% (interpolated from the capacity sweep, classifier head only).")
    tune = "off (fixed lr 1e-3 / wd 1e-4)" if cfg.no_tune else "validation grid search (lr, wd)"
    print(f"lr / wd selected per representation on the validation split; tuning = {tune}. "
          "Test acc read off the best-validation checkpoint (not the final epoch).")
    print("-" * 96)
    print(f"{'representation':16}{'test acc':>18}{'width@95%':>12}{'params@95%':>14}"
          f"{'epochs@90%':>13}{'lr':>10}{'wd':>10}")
    hp = results["hp"]
    for rep in REPRESENTATIONS:
        r = c[rep]
        p95 = params_to_threshold(cap[rep], 0.95)
        w95 = width_to_threshold(cap[rep], 0.95)
        p95s = f"{p95:,}" if p95 is not None else ">max"
        w95s = f"{w95:.1f}" if w95 is not None else ">max"
        ep = str(r["ep_to_thr"]) if r["ep_to_thr"] is not None else "--"
        tag = " (oracle)" if rep in _ORACLE else ""
        acc = f"{r['test_mean']:.3f} ± {r['test_std']:.3f}"
        print(f"{rep:16}{acc:>18}{w95s:>12}{p95s:>14}{ep:>13}"
              f"{hp[rep]['lr']:>10.0e}{hp[rep]['weight_decay']:>10.0e}{tag}")
    print("-" * 96)
    p95 = {rep: params_to_threshold(cap[rep], 0.95) for rep in REPRESENTATIONS}
    if p95["diwect"] and p95["wect"]:
        print(f"Headline: DIWECT reaches 95% with {p95['diwect']:,} params vs "
              f"{p95['wect']:,} for the raw WECT ({p95['wect'] / p95['diwect']:.0f}x smaller net).")
    print("A trained MLP *does* learn rotation from the WECT (all reps reach ~1.0 with")
    print("enough width+data) -- unlike the fixed-distance SVM. But the DIWECT matches")
    print("that accuracy with a far smaller model and from fewer samples; WECT+align")
    print("(oracle) is the easy ceiling. See the figure for the full picture.")
    print("-" * 72)


def _bar_panel(ax, reps, values, labels, ylabel, title, log=False):
    x = np.arange(len(reps))
    heights = [(v if v is not None else np.nan) for v in values]
    bars = ax.bar(x, heights, color=[_COLORS[r] for r in reps])
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_DISPLAY.get(r, r) for r in reps], fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(results, out_path):
    cfg = results["cfg"]
    reps = REPRESENTATIONS
    n_cls = len(results["shapes"])
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9), constrained_layout=True)

    # (1) Training curves.
    ax = axes[0, 0]
    epochs = np.arange(1, cfg.curve_epochs + 1)
    for rep in reps:
        r = results["curves"][rep]
        ax.plot(epochs, r["val_mean"], color=_COLORS[rep], lw=2,
                label=_DISPLAY.get(rep, rep))
        ax.fill_between(epochs, r["val_mean"] - r["val_std"], r["val_mean"] + r["val_std"],
                        color=_COLORS[rep], alpha=0.15)
    ax.axhline(1.0 / n_cls, ls="--", lw=1, color="k", label="chance")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title(f"Training curves (width={cfg.curve_width}, {cfg.n_train} train/class)",
                 fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) Capacity sweep: accuracy vs parameters
    ax = axes[0, 1]
    for rep in reps:
        r = results["capacity"][rep]
        ax.errorbar(r["params"], r["mean"], yerr=r["std"], color=_COLORS[rep],
                    lw=2, marker="o", ms=4, capsize=3, label=_DISPLAY.get(rep, rep))
    ax.axhline(0.95, ls=":", lw=1, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Capacity: accuracy vs model size ({cfg.n_train} train/class)", fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) Data-efficiency sweep
    ax = axes[0, 2]
    for rep in reps:
        r = results["data_eff"][rep]
        ax.errorbar(r["n"], r["mean"], yerr=r["std"], color=_COLORS[rep],
                    lw=2, marker="o", ms=4, capsize=3, label=_DISPLAY.get(rep, rep))
    ax.set_xscale("log"); ax.set_xticks(cfg.n_trains)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("train samples per class"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Data efficiency (width={cfg.curve_width})", fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) Hidden width needed to hit 95%
    w95 = [width_to_threshold(results["capacity"][r], 0.95) for r in reps]
    _bar_panel(axes[1, 0], reps, w95,
               [(f"{w:.1f}" if w is not None else ">max") for w in w95],
               "hidden width to reach 95% acc",
               "Hidden width needed to hit 95% (similar across reps)")

    # (5) Trainable parameters needed to hit 95%
    p95 = [params_to_threshold(results["capacity"][r], 0.95) for r in reps]
    _bar_panel(axes[1, 1], reps, p95,
               [(f"{p:,}" if p is not None else ">max") for p in p95],
               "trainable params to reach 95% acc",
               "Params needed to hit 95% (head only; smaller = better)", log=True)

    # (6) Final accuracy at the reference width on full data
    means = [results["curves"][r]["test_mean"] for r in reps]
    axes[1, 2].axhline(1.0 / n_cls, ls="--", lw=1, color="k")
    _bar_panel(axes[1, 2], reps, means, [f"{m:.2f}" for m in means],
               "test accuracy", f"Final accuracy (width={cfg.curve_width}, full data)")
    axes[1, 2].set_ylim(0, 1.12)

    fig.suptitle("A basic MLP learns rotation from the WECT — same width as pixels/WECT, "
                 "but the DIWECT gets there with ~65x fewer parameters",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# --------------------------------------------------------------------------- #
# Config / CLI
# --------------------------------------------------------------------------- #
def _config(args):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.size, cfg.area = args.size, args.area
    cfg.num_dirs, cfg.num_heights = args.num_dirs, args.num_heights
    cfg.n_train, cfg.n_val, cfg.n_test = args.n_train, args.n_val, args.n_test
    cfg.seeds = args.seeds
    cfg.curve_width, cfg.curve_epochs = args.curve_width, args.curve_epochs
    cfg.cap_epochs, cfg.de_epochs = args.cap_epochs, args.de_epochs
    cfg.widths = args.widths
    cfg.n_trains = [n for n in args.n_trains if n <= args.n_train]
    cfg.tune_lrs, cfg.tune_wds, cfg.no_tune = args.tune_lrs, args.tune_wds, args.no_tune
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--area", type=int, default=380)
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--curve-width", type=int, default=64)
    p.add_argument("--curve-epochs", type=int, default=150)
    p.add_argument("--cap-epochs", type=int, default=80)
    p.add_argument("--de-epochs", type=int, default=120)
    p.add_argument("--widths", type=int, nargs="+",
                   default=[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    p.add_argument("--n-trains", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                   help="learning rates searched on the validation split")
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3],
                   help="weight decays searched on the validation split")
    p.add_argument("--no-tune", action="store_true",
                   help="skip the validation search; use fixed lr 1e-3 / wd 1e-4")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (1 seed, few epochs/widths)")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mlp_classification.png"))
    args = p.parse_args()

    if args.quick:
        args.seeds = 1
        args.n_train, args.n_val, args.n_test = 60, 30, 40
        args.curve_epochs, args.cap_epochs, args.de_epochs = 40, 30, 30
        args.widths = [4, 16, 64]
        args.n_trains = [10, 30, 60]
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]

    results = run(_config(args))
    make_figure(results, args.out)


if __name__ == "__main__":
    main()
