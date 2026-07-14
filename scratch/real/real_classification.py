from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score

from core.wect import WECTTransform
from core.shapes import rotate_image
from core.mlp import train_mlp, predict, tune_head
from core.backbone import FrozenBackbone
from core.datasets import get_dataset, to_wect_image

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
REPRESENTATIONS = ["pixels", "wect", "diwect", "pixels+wect", "pixels+diwect"]
_COLORS = {"pixels": "#8d99ae", "wect": "#d1495b", "diwect": "#30638e",
           "pixels+wect": "#9b5de5", "pixels+diwect": "#f4a261"}
_DISPLAY = {"pixels+wect": "pix+WECT", "pixels+diwect": "pix+DIWECT"}


# --------------------------------------------------------------------------- #
# Features (extract once, cache, reuse)
# --------------------------------------------------------------------------- #
def build_features(cfg):
    ds = get_dataset(cfg.dataset, CACHE_DIR, cfg.wect_size)
    bgp, wsize, n_classes = ds["bg_percentile"], ds["wect_size"], ds["n_classes"]
    key = (f"feats_{cfg.dataset}_{cfg.backbone}_d{cfg.num_dirs}_h{cfg.num_heights}"
           f"_s{wsize}_bg{int(bgp)}.npz")
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        z = np.load(path)
        print(f"[loaded cached features {key}]")
        if ds["close"]:
            ds["close"]()
        base = {"pixels": z["pixels"], "wect": z["wect"], "diwect": z["diwect"]}
        return base, z["labels"], (z["tr"], z["va"], z["te"]), int(z["n_classes"])

    images, labels = ds["images"], ds["labels"]
    tr, va, te = ds["split"]
    n = len(labels)
    print(f"[{cfg.dataset}: extracting features for {n} images ...]")

    t0 = time.time()
    bb = FrozenBackbone(cfg.backbone)
    pixels = bb.extract_features(images, batch_size=128, progress=True)
    print(f"  backbone done in {time.time() - t0:.0f}s (dim {pixels.shape[1]}, {bb.device})")

    t0 = time.time()
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights)
    wect = np.empty((n, cfg.num_dirs * cfg.num_heights), np.float32)
    diwect = np.empty((n, cfg.num_heights), np.float32)
    for i in range(0, n, 256):
        batch = np.asarray(images[i:i + 256])
        for j, img in enumerate(batch):
            wi = to_wect_image(img, wsize, bgp)
            wect[i + j] = T.wect(wi).ravel()
            diwect[i + j] = T.diwect(wi)
        if i % 2560 == 0:
            print(f"    wect {i}/{n}", flush=True)
    print(f"  wect/diwect done in {time.time() - t0:.0f}s")
    if ds["close"]:
        ds["close"]()

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, pixels=pixels, wect=wect, diwect=diwect, labels=labels,
             tr=tr, va=va, te=te, n_classes=n_classes)
    print(f"[cached -> {key}]")
    return {"pixels": pixels, "wect": wect, "diwect": diwect}, labels, (tr, va, te), n_classes


def assemble(base, rep, idx=None):
    def take(name):
        return base[name] if idx is None else base[name][idx]
    if rep in base:
        return take(rep)
    return np.concatenate([take(p) for p in rep.split("+")], axis=1)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def _splits(base, labels, tr, va, te):
    return {rep: (assemble(base, rep, tr), assemble(base, rep, va), assemble(base, rep, te))
            for rep in REPRESENTATIONS}


def tune_hyperparams(feats, y, tr, va, te, cfg):
    if cfg.no_tune:
        return {rep: {"lr": 1e-3, "weight_decay": 1e-4, "val_acc": None}
                for rep in REPRESENTATIONS}
    grid = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    hp = {}
    for rep in REPRESENTATIONS:
        Xtr, Xva, Xte = feats[rep]
        best = tune_head(Xtr, y[tr], Xva, y[va], Xte, y[te],
                         width=cfg.head_width, epochs=cfg.epochs, seed=0, grid=grid)
        hp[rep] = {"lr": best["hp"]["lr"], "weight_decay": best["hp"]["weight_decay"],
                   "val_acc": best["best_val_acc"]}
        print(f"    tuned {rep:14} lr={hp[rep]['lr']:.0e} wd={hp[rep]['weight_decay']:.0e}"
              f"  (val acc {hp[rep]['val_acc']:.3f})", flush=True)
    return hp


def run_head_to_head(feats, y, tr, va, te, cfg, hp):
    out = {}
    for rep in REPRESENTATIONS:
        Xtr, Xva, Xte = feats[rep]
        accs, f1s = [], []
        for seed in range(cfg.seeds):
            r = train_mlp(Xtr, y[tr], Xva, y[va], Xte, y[te],
                          width=cfg.head_width, epochs=cfg.epochs, seed=seed,
                          lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
            accs.append(r["test_acc"])
            f1s.append(f1_score(y[te], r["test_pred"], average="macro"))
        out[rep] = {"acc": np.array(accs), "f1": np.array(f1s), "params": r["params"],
                    "hp": hp[rep]}
    return out


def run_data_efficiency(feats, y, tr, va, te, cfg, hp):
    out = {rep: {"n": [], "mean": [], "std": []} for rep in REPRESENTATIONS}
    for frac in cfg.fracs:
        for seed in range(cfg.seeds):
            rng = np.random.default_rng(1000 * seed + int(frac * 1000))
            sub = _stratified_subsample(y[tr], frac, rng)
            for rep in REPRESENTATIONS:
                Xtr, Xva, Xte = feats[rep]
                r = train_mlp(Xtr[sub], y[tr][sub], Xva, y[va], Xte, y[te],
                              width=cfg.head_width, epochs=cfg.epochs, seed=seed,
                              lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
                out[rep].setdefault("_tmp", {}).setdefault(frac, []).append(r["test_acc"])
    n_tr = len(tr)
    for rep in REPRESENTATIONS:
        for frac in cfg.fracs:
            vals = out[rep]["_tmp"][frac]
            out[rep]["n"].append(int(round(frac * n_tr)))
            out[rep]["mean"].append(float(np.mean(vals)))
            out[rep]["std"].append(float(np.std(vals)))
        del out[rep]["_tmp"]
    return out


def _stratified_subsample(y, frac, rng):
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        k = max(1, int(round(frac * len(pool))))
        idx.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(idx)


def run_capacity(feats, y, tr, va, te, cfg, hp):
    out = {rep: {"widths": cfg.widths, "params": [], "mean": [], "std": []}
           for rep in REPRESENTATIONS}
    for rep in REPRESENTATIONS:
        Xtr, Xva, Xte = feats[rep]
        for w in cfg.widths:
            accs, params = [], None
            for seed in range(cfg.seeds):
                r = train_mlp(Xtr, y[tr], Xva, y[va], Xte, y[te],
                              width=w, epochs=cfg.cap_epochs, seed=seed,
                              lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
                accs.append(r["test_acc"])
                params = r["params"]
            out[rep]["mean"].append(float(np.mean(accs)))
            out[rep]["std"].append(float(np.std(accs)))
            out[rep]["params"].append(params)
    return out


def _rotate_rgb(rgb_uint8, angle):
    if angle == 0:
        return rgb_uint8
    x = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).float()
    x = TF.rotate(x, float(angle), interpolation=TF.InterpolationMode.BILINEAR)
    return x.clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).numpy()


def _center_crop(rgb_uint8, size):
    h = rgb_uint8.shape[1]
    o = (h - size) // 2
    return rgb_uint8[:, o:o + size, o:o + size]


def run_rotation(base, y, tr, va, te, cfg, hp):
    heads = {}
    for rep in REPRESENTATIONS:
        Xtr, Xva, Xte = (assemble(base, rep, tr), assemble(base, rep, va), assemble(base, rep, te))
        r = train_mlp(Xtr, y[tr], Xva, y[va], Xte, y[te],
                      width=cfg.head_width, epochs=cfg.epochs, seed=0,
                      lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
        heads[rep] = (r["model"], r["scaler"])

    # Rotation test subset
    rng = np.random.default_rng(0)
    sub = _stratified_subsample(y[te], min(1.0, cfg.rot_n / len(te)), rng)
    te_sub = np.sort(te[sub])
    ds = get_dataset(cfg.dataset, CACHE_DIR, cfg.wect_size)
    rgb = np.asarray(ds["images"][te_sub])    # (m,H,W,3)
    if ds["close"]:
        ds["close"]()
    y_sub = y[te_sub]
    base_wimgs = [to_wect_image(im, ds["wect_size"], ds["bg_percentile"]) for im in rgb]

    bb = FrozenBackbone(cfg.backbone)
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights)

    crop = int(rgb.shape[1] / np.sqrt(2))  # inscribed square: valid at any rotation
    out = {rep: {"angles": cfg.angles, "acc": []} for rep in REPRESENTATIONS}
    for ang in cfg.angles:
        pix = bb.extract_features(_center_crop(_rotate_rgb(rgb, ang), crop), batch_size=128)
        m = len(y_sub)
        wect = np.empty((m, cfg.num_dirs * cfg.num_heights), np.float32)
        diw = np.empty((m, cfg.num_heights), np.float32)
        for j, wi in enumerate(base_wimgs):
            wr = wi if ang == 0 else rotate_image(wi, ang)
            wect[j] = T.wect(wr).ravel()
            diw[j] = T.diwect(wr)
        rot_base = {"pixels": pix, "wect": wect, "diwect": diw}
        for rep in REPRESENTATIONS:
            model, scaler = heads[rep]
            pred = predict(model, scaler, assemble(rot_base, rep))
            out[rep]["acc"].append(float(np.mean(pred == y_sub)))
    return out


# --------------------------------------------------------------------------- #
# Report + figure
# --------------------------------------------------------------------------- #
def _report(results):
    cfg = results["cfg"]
    h2h = results["head_to_head"]
    print(f"\n{cfg.dataset} | {results['n_classes']} classes | frozen {cfg.backbone} + WECT/DIWECT "
          f"| head width {cfg.head_width}, {cfg.seeds} seeds")
    print("head params = trainable params of the MLP head at the reference width "
          "(the WECT transform / frozen backbone add NO trainable params).")
    tune = "off (fixed lr 1e-3 / wd 1e-4)" if cfg.no_tune else "validation grid search (lr, wd)"
    print(f"lr / wd selected per representation on the validation split; tuning = {tune}. "
          "Model = best-validation-accuracy checkpoint (not final epoch).")
    print("-" * 90)
    print(f"{'representation':16}{'test acc':>16}{'macro-F1':>16}{'head params':>16}{'lr':>12}{'wd':>12}")
    for rep in REPRESENTATIONS:
        a, ff = h2h[rep]["acc"], h2h[rep]["f1"]
        hp = h2h[rep]["hp"]
        print(f"{_DISPLAY.get(rep, rep):16}{f'{a.mean():.3f} ± {a.std():.3f}':>16}"
              f"{f'{ff.mean():.3f} ± {ff.std():.3f}':>16}{h2h[rep]['params']:>16,}"
              f"{hp['lr']:>12.0e}{hp['weight_decay']:>12.0e}")
    print("-" * 90)
    d, w = h2h["diwect"]["acc"].mean(), h2h["wect"]["acc"].mean()
    pd, pw = h2h["pixels+diwect"]["acc"].mean(), h2h["pixels+wect"]["acc"].mean()
    px = h2h["pixels"]["acc"].mean()
    print(f"DIWECT vs WECT: {d:.3f} vs {w:.3f} ({d - w:+.3f}).  "
          f"pixels {px:.3f} -> pix+DIWECT {pd:.3f} ({pd - px:+.3f}), "
          f"pix+WECT {pw:.3f} ({pw - px:+.3f}).")
    rot = results["rotation"]
    worst = lambda r: float(np.min(rot[r]["acc"]))
    print(f"Rotation (worst-angle test acc; higher = more rotation-stable): "
          f"DIWECT {worst('diwect'):.3f} vs WECT {worst('wect'):.3f}; "
          f"pix+DIWECT {worst('pixels+diwect'):.3f} vs pix+WECT {worst('pixels+wect'):.3f}.")
    print("-" * 78)


def make_figure(results, out_path):
    cfg = results["cfg"]
    reps = REPRESENTATIONS
    h2h = results["head_to_head"]
    best = max(h2h[r]["acc"].mean() for r in reps)
    thr = cfg.thr_frac * best
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)

    # (1) Head-to-head: accuracy + macro-F1 grouped bars
    ax = axes[0, 0]
    x = np.arange(len(reps))
    accm = [h2h[r]["acc"].mean() for r in reps]
    acce = [h2h[r]["acc"].std() for r in reps]
    f1m = [h2h[r]["f1"].mean() for r in reps]
    ax.bar(x - 0.2, accm, 0.4, yerr=acce, capsize=3,
           color=[_COLORS[r] for r in reps], label="accuracy")
    ax.bar(x + 0.2, f1m, 0.4, color=[_COLORS[r] for r in reps], alpha=0.5, label="macro-F1")
    ax.axhline(1.0 / results["n_classes"], ls="--", lw=1, color="k", label="chance")
    ax.set_xticks(x); ax.set_xticklabels([_DISPLAY.get(r, r) for r in reps], fontsize=8, rotation=15)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.set_title("Head-to-head (full data): accuracy & macro-F1", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (2) Data efficiency
    ax = axes[0, 1]
    for rep in reps:
        r = results["data_eff"][rep]
        ax.errorbar(r["n"], r["mean"], yerr=r["std"], color=_COLORS[rep], lw=2,
                    marker="o", ms=4, capsize=3, label=_DISPLAY.get(rep, rep))
    ax.set_xscale("log"); ax.set_xlabel("train samples"); ax.set_ylabel("test accuracy")
    ax.set_title("Data efficiency", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (3) Capacity: accuracy vs head parameters
    ax = axes[1, 0]
    for rep in reps:
        r = results["capacity"][rep]
        ax.errorbar(r["params"], r["mean"], yerr=r["std"], color=_COLORS[rep], lw=2,
                    marker="o", ms=4, capsize=3, label=_DISPLAY.get(rep, rep))
    ax.axhline(thr, ls=":", lw=1, color="gray", label=f"{cfg.thr_frac:.0%} of best")
    ax.set_xscale("log"); ax.set_xlabel("head parameters (backbone/WECT not counted)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Capacity: accuracy vs head size", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (4) Rotation robustness
    ax = axes[1, 1]
    for rep in reps:
        r = results["rotation"][rep]
        ax.plot(r["angles"], r["acc"], color=_COLORS[rep], lw=2, marker="o", ms=4,
                label=_DISPLAY.get(rep, rep))
    ax.set_xlabel("test-set rotation (degrees)"); ax.set_ylabel("test accuracy")
    ax.set_title("Rotation robustness of the test set (center-cropped)", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{cfg.dataset}: does the DIWECT match the WECT and keep its efficiency / "
                 "rotation-robustness on real images?", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(cfg):
    base, labels, (tr, va, te), n_classes = build_features(cfg)
    if cfg.max_train and len(tr) > cfg.max_train:
        # split indices are ordered class-by-class; shuffle before capping so the
        # subset stays representative of all classes
        tr = np.random.default_rng(cfg.split_seed).permutation(tr)[:cfg.max_train]
    print(f"[split: {len(tr)} train / {len(va)} val / {len(te)} test | {n_classes} classes]")
    feats = _splits(base, labels, tr, va, te)

    results = {"cfg": cfg, "n_classes": n_classes}
    t = time.time()
    tag = "held-fixed defaults (--no-tune)" if cfg.no_tune else "validation grid search"
    print(f"[0/4] hyperparameter tuning ({tag}) ...", flush=True)
    hp = tune_hyperparams(feats, labels, tr, va, te, cfg)
    results["hp"] = hp
    print(f"[1/4] head-to-head ... ({time.time() - t:.0f}s)", flush=True)
    results["head_to_head"] = run_head_to_head(feats, labels, tr, va, te, cfg, hp)
    print(f"[2/4] data efficiency ... ({time.time() - t:.0f}s)", flush=True)
    results["data_eff"] = run_data_efficiency(feats, labels, tr, va, te, cfg, hp)
    print(f"[3/4] capacity sweep ... ({time.time() - t:.0f}s)", flush=True)
    results["capacity"] = run_capacity(feats, labels, tr, va, te, cfg, hp)
    print(f"[4/4] rotation robustness ... ({time.time() - t:.0f}s)", flush=True)
    results["rotation"] = run_rotation(base, labels, tr, va, te, cfg, hp)
    print(f"[done sweeps in {time.time() - t:.0f}s]", flush=True)
    _report(results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="galaxy10",
                   choices=["galaxy10", "bloodmnist", "organamnist"])
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--wect-size", type=int, default=64)
    p.add_argument("--head-width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--cap-epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--angles", type=int, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--rot-n", type=int, default=1500)
    p.add_argument("--thr-frac", type=float, default=0.90)
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                   help="learning rates searched on the validation split")
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3],
                   help="weight decays searched on the validation split")
    p.add_argument("--no-tune", action="store_true",
                   help="skip the validation search; use fixed lr 1e-3 / wd 1e-4")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0, help="cap train size (0 = all)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"{args.dataset}_classification.png")

    if args.quick:
        args.seeds = 1
        args.epochs, args.cap_epochs = 40, 30
        args.widths = [4, 16, 64]
        args.fracs = [0.05, 0.25, 1.0]
        args.angles = [0, 90, 180, 270]
        args.rot_n = 400
        args.max_train = 2000
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]

    run_cfg = args
    results = run(run_cfg)
    make_figure(results, args.out)


if __name__ == "__main__":
    main()
