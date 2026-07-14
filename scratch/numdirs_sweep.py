from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from core.wect import WECTTransform
from core.shapes import make_shape_library, rotate_image
from core.mlp import train_mlp, tune_head
from synthetic.classification import svm_classification as svmc
from synthetic.classification import mlp_classification as mlpc
from real import real_classification as rc

HERE = os.path.dirname(os.path.abspath(__file__))

CSV_COLUMNS = ["experiment", "dataset", "classifier", "representation", "num_dirs",
               "num_heights", "feature_dim", "head_params", "seed", "accuracy", "macro_f1"]


# --------------------------------------------------------------------------- #
# Direction subsampling
# --------------------------------------------------------------------------- #
def subsample_dirs(wect_flat, n_full, num_heights, d):
    if n_full % d:
        raise ValueError(f"num_dirs {d} must divide max_dirs {n_full}")
    w = wect_flat.reshape(-1, n_full, num_heights)[:, :: n_full // d, :]
    return (w.reshape(w.shape[0], d * num_heights).astype(np.float32),
            w.mean(1).astype(np.float32))


def _tuned_hp(Xtr, ytr, Xva, yva, Xte, yte, width, epochs, grid):
    best = tune_head(Xtr, ytr, Xva, yva, Xte, yte, width=width, epochs=epochs,
                     seed=0, grid=grid)
    return best["hp"]["lr"], best["hp"]["weight_decay"]


# --------------------------------------------------------------------------- #
# Synthetic distance-kernel SVM on equal-area shapes
# --------------------------------------------------------------------------- #
def run_svm(writer, cfg):
    shapes = make_shape_library(size=cfg.size, area=cfg.area)
    T = WECTTransform(num_dirs=cfg.max_dirs, num_heights=cfg.num_heights)
    H, N = cfg.num_heights, cfg.max_dirs
    stack = lambda imgs: np.stack([T.wect(im).ravel() for im in imgs]).astype(np.float32)

    for seed in range(cfg.svm_seeds):
        tr_imgs, ytr, atr = svmc.make_dataset(shapes, cfg.svm_train, seed)
        te_imgs, yte, ate = svmc.make_dataset(shapes, cfg.svm_test, 1000 + seed)
        Wtr, Wte = stack(tr_imgs), stack(te_imgs)
        WAtr = stack([rotate_image(im, -a) for im, a in zip(tr_imgs, atr)])
        WAte = stack([rotate_image(im, -a) for im, a in zip(te_imgs, ate)])
        for d in cfg.grid:
            wtr, dtr = subsample_dirs(Wtr, N, H, d)
            wte, dte = subsample_dirs(Wte, N, H, d)
            watr, _ = subsample_dirs(WAtr, N, H, d)
            wate, _ = subsample_dirs(WAte, N, H, d)
            reps = {"wect": (wtr, wte, d * H), "wect_aligned": (watr, wate, d * H),
                    "diwect": (dtr, dte, H)}
            for rep, (Xtr, Xte, fdim) in reps.items():
                pred = svmc.svm_predict(Xtr, ytr, Xte, C=cfg.svm_C)
                writer.row("svm", "shapes", "svm", rep, d, H, fdim, "", seed,
                           accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))


# --------------------------------------------------------------------------- #
# Synthetic one-hidden-layer MLP on equal-area shapes
# --------------------------------------------------------------------------- #
def run_mlp_shapes(writer, cfg):
    shapes = make_shape_library(size=cfg.size, area=cfg.area)
    T = WECTTransform(num_dirs=cfg.max_dirs, num_heights=cfg.num_heights)
    H, N = cfg.num_heights, cfg.max_dirs
    stack = lambda imgs: np.stack([T.wect(im).ravel() for im in imgs]).astype(np.float32)
    grid_hp = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}

    # Featurize each seed's train/val/test at max_dirs (raw + ground-truth-aligned).
    pool = {}
    for seed in range(cfg.mlp_seeds):
        parts = {}
        for nm, npc, sd in [("tr", cfg.mlp_train, 10 * seed + 1),
                            ("va", cfg.mlp_val, 10 * seed + 2),
                            ("te", cfg.mlp_test, 10 * seed + 3)]:
            imgs, y, ang = mlpc.make_dataset(shapes, npc, sd)
            aligned = [rotate_image(im, -a) for im, a in zip(imgs, ang)]
            parts[nm] = (stack(imgs), stack(aligned), y)
        pool[seed] = parts

    reps = ["wect", "wect_aligned", "diwect"]
    for rep in reps:
        # Tune lr/wd once at max_dirs on seed 0, then hold across the D sweep.
        (Wtr, WAtr, ytr) = pool[0]["tr"]; (Wva, WAva, yva) = pool[0]["va"]
        (Wte, WAte, yte) = pool[0]["te"]
        src0 = {"wect": (Wtr, Wva, Wte), "wect_aligned": (WAtr, WAva, WAte),
                "diwect": (Wtr, Wva, Wte)}[rep]
        lr, wd = _tuned_hp(src0[0], ytr, src0[1], yva, src0[2], yte,
                           cfg.mlp_width, cfg.mlp_epochs, grid_hp)
        for d in cfg.grid:
            for seed in range(cfg.mlp_seeds):
                (Wtr, WAtr, ytr) = pool[seed]["tr"]; (Wva, WAva, yva) = pool[seed]["va"]
                (Wte, WAte, yte) = pool[seed]["te"]
                def feats(Wf, Waf):
                    w, dw = subsample_dirs(Wf, N, H, d)
                    wa, _ = subsample_dirs(Waf, N, H, d)
                    return {"wect": w, "wect_aligned": wa, "diwect": dw}
                Xtr = feats(Wtr, WAtr)[rep]; Xva = feats(Wva, WAva)[rep]
                Xte = feats(Wte, WAte)[rep]
                r = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, width=cfg.mlp_width,
                              epochs=cfg.mlp_epochs, seed=seed, lr=lr, weight_decay=wd)
                writer.row("mlp", "shapes", "mlp", rep, d, H, Xtr.shape[1],
                           r["params"], seed, r["test_acc"],
                           f1_score(yte, r["test_pred"], average="macro"))


# --------------------------------------------------------------------------- #
# Real images frozen-backbone + WECT/DIWECT
# --------------------------------------------------------------------------- #
def run_real(writer, dataset, cfg):
    class C:
        pass
    fc = C()
    fc.dataset, fc.backbone = dataset, cfg.backbone
    fc.num_dirs, fc.num_heights, fc.wect_size = cfg.max_dirs, cfg.num_heights, cfg.wect_size
    base, labels, (tr, va, te), n_classes = rc.build_features(fc)
    pixels, wect_full = base["pixels"], base["wect"]
    H, N = cfg.num_heights, cfg.max_dirs
    grid_hp = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}

    def assemble(rep, wect_d, diw_d):
        parts = {"pixels": pixels, "wect": wect_d, "diwect": diw_d}
        return np.concatenate([parts[p] for p in rep.split("+")], axis=1)

    reps = ["wect", "diwect", "pixels+wect", "pixels+diwect"]
    # Precompute the full-direction (max_dirs) features per rep for HP tuning.
    w_max, dw_max = subsample_dirs(wect_full, N, H, N)
    for rep in reps:
        Xrep = assemble(rep, w_max, dw_max)
        lr, wd = _tuned_hp(Xrep[tr], labels[tr], Xrep[va], labels[va],
                           Xrep[te], labels[te], cfg.real_width, cfg.real_epochs, grid_hp)
        for d in cfg.grid:
            wd_, dw_ = subsample_dirs(wect_full, N, H, d)
            X = assemble(rep, wd_, dw_)
            for seed in range(cfg.real_seeds):
                r = train_mlp(X[tr], labels[tr], X[va], labels[va], X[te], labels[te],
                              width=cfg.real_width, epochs=cfg.real_epochs, seed=seed,
                              lr=lr, weight_decay=wd)
                writer.row("real", dataset, "mlp", rep, d, H, X.shape[1], r["params"],
                           seed, r["test_acc"],
                           f1_score(labels[te], r["test_pred"], average="macro"))


# --------------------------------------------------------------------------- #
# Pixels-only baseline
# --------------------------------------------------------------------------- #
def run_pixels(writer, exp, dataset, cfg):
    H = cfg.num_heights
    grid_hp = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    if exp == "svm":
        shapes = make_shape_library(size=cfg.size, area=cfg.area)
        fdim = cfg.size * cfg.size
        for seed in range(cfg.svm_seeds):
            tr_imgs, ytr, _ = svmc.make_dataset(shapes, cfg.svm_train, seed)
            te_imgs, yte, _ = svmc.make_dataset(shapes, cfg.svm_test, 1000 + seed)
            Xtr = np.stack([im.ravel() for im in tr_imgs]).astype(np.float32)
            Xte = np.stack([im.ravel() for im in te_imgs]).astype(np.float32)
            pred = svmc.svm_predict(Xtr, ytr, Xte, C=cfg.svm_C)
            writer.row("svm", "shapes", "svm", "pixels", 0, H, fdim, "", seed,
                       accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
    elif exp == "mlp":
        shapes = make_shape_library(size=cfg.size, area=cfg.area)
        pool = {}
        for seed in range(cfg.mlp_seeds):
            parts = {}
            for nm, npc, sd in [("tr", cfg.mlp_train, 10 * seed + 1),
                                ("va", cfg.mlp_val, 10 * seed + 2),
                                ("te", cfg.mlp_test, 10 * seed + 3)]:
                imgs, y, _ = mlpc.make_dataset(shapes, npc, sd)
                parts[nm] = (np.stack([im.ravel() for im in imgs]).astype(np.float32), y)
            pool[seed] = parts
        (Xtr, ytr) = pool[0]["tr"]; (Xva, yva) = pool[0]["va"]; (Xte, yte) = pool[0]["te"]
        lr, wd = _tuned_hp(Xtr, ytr, Xva, yva, Xte, yte, cfg.mlp_width, cfg.mlp_epochs, grid_hp)
        for seed in range(cfg.mlp_seeds):
            (Xtr, ytr) = pool[seed]["tr"]; (Xva, yva) = pool[seed]["va"]; (Xte, yte) = pool[seed]["te"]
            r = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, width=cfg.mlp_width,
                          epochs=cfg.mlp_epochs, seed=seed, lr=lr, weight_decay=wd)
            writer.row("mlp", "shapes", "mlp", "pixels", 0, H, Xtr.shape[1], r["params"],
                       seed, r["test_acc"], f1_score(yte, r["test_pred"], average="macro"))
    else:  # real
        class C:
            pass
        fc = C()
        fc.dataset, fc.backbone = dataset, cfg.backbone
        fc.num_dirs, fc.num_heights, fc.wect_size = cfg.max_dirs, H, cfg.wect_size
        base, labels, (tr, va, te), n_classes = rc.build_features(fc)
        X = base["pixels"]
        lr, wd = _tuned_hp(X[tr], labels[tr], X[va], labels[va], X[te], labels[te],
                           cfg.real_width, cfg.real_epochs, grid_hp)
        for seed in range(cfg.real_seeds):
            r = train_mlp(X[tr], labels[tr], X[va], labels[va], X[te], labels[te],
                          width=cfg.real_width, epochs=cfg.real_epochs, seed=seed,
                          lr=lr, weight_decay=wd)
            writer.row("real", dataset, "mlp", "pixels", 0, H, X.shape[1], r["params"],
                       seed, r["test_acc"], f1_score(labels[te], r["test_pred"], average="macro"))


# --------------------------------------------------------------------------- #
# CSV writer + driver
# --------------------------------------------------------------------------- #
class CsvWriter:

    def __init__(self, path, resume=False):
        self.path = path
        new = not (resume and os.path.exists(path))
        self.f = open(path, "a" if resume else "w", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(CSV_COLUMNS)
            self.f.flush()

    def row(self, *vals):
        out = []
        for v in vals:
            out.append(f"{v:.6f}" if isinstance(v, float) else v)
        self.w.writerow(out)
        self.f.flush()

    def close(self):
        self.f.close()


EXP_MAP = {"svm": ("svm", "shapes"), "mlp": ("mlp", "shapes"),
           "bloodmnist": ("real", "bloodmnist"), "galaxy10": ("real", "galaxy10")}


def run(cfg):
    writer = CsvWriter(cfg.out, resume=cfg.resume)
    dispatch = {
        "svm": lambda: run_svm(writer, cfg),
        "mlp": lambda: run_mlp_shapes(writer, cfg),
        "bloodmnist": lambda: run_real(writer, "bloodmnist", cfg),
        "galaxy10": lambda: run_real(writer, "galaxy10", cfg),
    }
    try:
        for name in cfg.experiments:
            t = time.time()
            if cfg.pixels_only:
                exp, dset = EXP_MAP[name]
                print(f"[{name}] pixels-only baseline ...", flush=True)
                run_pixels(writer, exp, dset, cfg)
            else:
                print(f"[{name}] num_dirs sweep over {cfg.grid} ...", flush=True)
                dispatch[name]()
            print(f"[{name}] done in {time.time() - t:.0f}s -> {cfg.out}", flush=True)
    finally:
        writer.close()
    kind = "pixels baselines" if cfg.pixels_only else "sweeps"
    print(f"\nAll requested {kind} written to {cfg.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+",
                   choices=["svm", "mlp", "bloodmnist", "galaxy10"],
                   default=["svm", "mlp", "bloodmnist", "galaxy10"])
    p.add_argument("--grid", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                   help="num_dirs values (each must divide --max-dirs)")
    p.add_argument("--max-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--wect-size", type=int, default=64)
    p.add_argument("--backbone", default="resnet18")
    # tuning grid (held fixed across the num_dirs sweep, selected on validation)
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    # synthetic shape settings
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--area", type=int, default=380)
    p.add_argument("--svm-train", type=int, default=6)
    p.add_argument("--svm-test", type=int, default=40)
    p.add_argument("--svm-seeds", type=int, default=6)
    p.add_argument("--svm-C", type=float, default=10.0)
    p.add_argument("--mlp-train", type=int, default=200)
    p.add_argument("--mlp-val", type=int, default=50)
    p.add_argument("--mlp-test", type=int, default=100)
    p.add_argument("--mlp-seeds", type=int, default=3)
    p.add_argument("--mlp-width", type=int, default=64)
    p.add_argument("--mlp-epochs", type=int, default=120)
    # real settings
    p.add_argument("--real-seeds", type=int, default=3)
    p.add_argument("--real-width", type=int, default=64)
    p.add_argument("--real-epochs", type=int, default=60)
    p.add_argument("--pixels-only", action="store_true",
                   help="compute only the pixels baseline per experiment (num_dirs=0 rows)")
    p.add_argument("--resume", action="store_true",
                   help="append to an existing CSV instead of overwriting")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "results", "numdirs_sweep.csv"))
    args = p.parse_args()

    if args.quick:
        args.grid = [1, 8, 64]
        args.svm_seeds = 2
        args.mlp_seeds, args.mlp_train, args.mlp_val, args.mlp_test = 1, 60, 30, 40
        args.mlp_epochs = 40
        args.real_seeds, args.real_epochs = 1, 25
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]

    for d in args.grid:
        if args.max_dirs % d:
            p.error(f"--grid value {d} does not divide --max-dirs {args.max_dirs}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
