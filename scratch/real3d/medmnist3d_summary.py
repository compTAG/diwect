"""One-figure synthesis of the MedMNIST3D experiments: where the DIWECT wins.

A 2x3 figure. Rows are the two classifiers -- a fixed-kernel **SVM** (top) and a
flexible **MLP** (bottom); columns are the three regimes that matter for the
direction-integrated DIWECT vs the raw WECT / voxels:

  1. Native pose (test == training orientation). The directional WECT edges
     ahead -- the price the DIWECT pays for averaging orientation away.

  2. SO(3)-rotated test. Two bars per representation: train native -> test
     rotated, and train SO(3) -> test SO(3) (rotation augmentation); the dotted
     line marks the native-pose level. The WECT and voxels drop below it and
     only partly recover with augmentation, while the DIWECT sits on its native
     line -- it is invariant, so orientation (and augmentation) do nothing. The
     flexible MLP overfits the training pose worst, so its WECT falls furthest.

  3. Efficiency. Feature dimension (48 for DIWECT vs 3072 for the WECT vs 21952
     voxels) annotated with the rotated macro-F1: the DIWECT is the smallest
     feature *and* the best under rotation.

Averaged over adrenalmnist3d and vesselmnist3d -- the two MedMNIST3D sets that
ship as genuine binary shape masks (synapse/nodule/organ/fracture are grayscale
intensity volumes, a different task). Rotations are exact on the mesh; training
is minority-oversampled and scored by macro-F1 (the sets are imbalanced).
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from core.wect import WECTTransform
from core.mlp import train_mlp, tune_head
from core.meshes import (occ_to_mesh, mesh_to_complex, voxelize,
                         random_rotation, rotate)
from core.medmnist3d import load_medmnist3d, volume_to_occupancy

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_HALF = 1.75
DATASETS = ["adrenalmnist3d", "vesselmnist3d"]  # the genuine binary shape-mask sets
REPS = ["voxels", "wect", "diwect"]
CONDITIONS = [("native", "native"), ("native", "rot"), ("rot", "rot")]
_COLORS = {"voxels": "#8d99ae", "wect": "#d1495b", "diwect": "#30638e"}
_CSV = ["dataset", "classifier", "rep", "train_mode", "test_mode", "seed",
        "macro_f1", "head_params", "feat_dim"]


def svm_predict(Xtr, ytr, Xte, C=10.0):
    d2 = cdist(Xtr, Xtr, "sqeuclidean")
    gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-12)
    clf = SVC(kernel="precomputed", C=C).fit(np.exp(-gamma * d2), ytr)
    return clf.predict(np.exp(-gamma * cdist(Xte, Xtr, "sqeuclidean")))


def balance_positions(y_sub, seed):
    rng = np.random.default_rng(seed)
    target = np.bincount(y_sub).max()
    out = []
    for c in np.unique(y_sub):
        pool = np.where(y_sub == c)[0]
        extra = rng.choice(pool, target - len(pool), replace=True) if len(pool) < target \
            else np.empty(0, int)
        out.append(np.concatenate([pool, extra]))
    return rng.permutation(np.concatenate(out))


def featurize(meshes, T, mode, seed):
    """WECT / DIWECT / voxels for a set of meshes under a native or SO(3) pose."""
    rng = np.random.default_rng(seed)
    W, D, V = [], [], []
    for m in meshes:
        v = rotate(m.verts, random_rotation(rng)) if mode == "rot" else m.verts
        mat = T.wect(mesh_to_complex(v, m.faces)).reshape(T.num_dirs, T.num_heights)
        W.append(mat.ravel())
        D.append(mat.mean(0))
        V.append(voxelize(v, m.faces, grid=28, half=WORLD_HALF).ravel())
    return {"wect": np.stack(W).astype(np.float32),
            "diwect": np.stack(D).astype(np.float32),
            "voxels": np.stack(V).astype(np.float32)}


def run_dataset(name, cfg, writer):
    ds = load_medmnist3d(name, CACHE_DIR)
    vols, y = ds["volumes"], ds["labels"]
    tr, va, te = ds["split"]
    t0 = time.time()
    meshes = [occ_to_mesh(volume_to_occupancy(v, ds["is_mask"], cfg.gray_percentile))
              for v in vols]
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights, dim=3)
    feats = {}
    rot_seed = {"tr": 701, "va": 702, "te": 703}
    for sp, ids in [("tr", tr), ("va", va), ("te", te)]:
        ms = [meshes[i] for i in ids]
        feats[sp] = {"native": featurize(ms, T, "native", 0),
                     "rot": featurize(ms, T, "rot", rot_seed[sp])}
    print(f"  [{name}] featurized in {time.time() - t0:.0f}s | balance {np.bincount(y)}", flush=True)

    grid = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    for rep in REPS:
        best = tune_head(feats["tr"]["native"][rep], y[tr], feats["va"]["native"][rep], y[va],
                         feats["te"]["native"][rep], y[te], width=cfg.width, epochs=cfg.epochs,
                         seed=0, grid=grid)
        lr, wd = best["hp"]["lr"], best["hp"]["weight_decay"]
        for tr_m, te_m in CONDITIONS:
            for seed in range(cfg.seeds):
                pos = balance_positions(y[tr], seed)
                Xtr, ytr = feats["tr"][tr_m][rep][pos], y[tr][pos]
                Xva, Xte = feats["va"][te_m][rep], feats["te"][te_m][rep]
                # MLP
                r = train_mlp(Xtr, ytr, Xva, y[va], Xte, y[te], width=cfg.width,
                              epochs=cfg.epochs, seed=seed, lr=lr, weight_decay=wd)
                writer.writerow([name, "mlp", rep, tr_m, te_m, seed,
                                 f1_score(y[te], r["test_pred"], average="macro"),
                                 r["params"], Xtr.shape[1]])
                # SVM
                pred = svm_predict(Xtr, ytr, Xte, C=cfg.svm_C)
                writer.writerow([name, "svm", rep, tr_m, te_m, seed,
                                 f1_score(y[te], pred, average="macro"), "", Xtr.shape[1]])
        print(f"    {rep:8} done", flush=True)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["macro_f1"] = float(r["macro_f1"])
        r["feat_dim"] = int(r["feat_dim"])
        r["head_params"] = int(r["head_params"]) if r["head_params"] else None
    return rows


def _agg(rows, **filt):
    vals = [r["macro_f1"] for r in rows
            if all(r[k] == v for k, v in filt.items())]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)


def _per_dataset(rows, **filt):
    out = []
    for d in DATASETS:
        v = [r["macro_f1"] for r in rows if r["dataset"] == d
             and all(r[k] == val for k, val in filt.items())]
        if v:
            out.append(np.mean(v))
    return out


def make_figure(csv_path, out_path):
    rows = _load(csv_path)
    feat_dim = {}
    params = {}
    for r in rows:
        feat_dim.setdefault(r["rep"], r["feat_dim"])
        if r["classifier"] == "mlp" and r["head_params"]:
            params.setdefault(r["rep"], r["head_params"])

    clf_rows = [("svm", "SVM  (fixed RBF kernel)"), ("mlp", "MLP  (flexible head)")]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), constrained_layout=True)
    x = np.arange(len(REPS))

    for row, (clf, clf_label) in enumerate(clf_rows):
        nat = [_agg(rows, classifier=clf, rep=r, train_mode="native", test_mode="native")[0] for r in REPS]
        rot = [_agg(rows, classifier=clf, rep=r, train_mode="native", test_mode="rot")[0] for r in REPS]

        # -- Col 0: native pose --
        ax = axes[row, 0]
        b = ax.bar(x, nat, 0.62, color=[_COLORS[r] for r in REPS])
        ax.bar_label(b, labels=[f"{v:.2f}" for v in nat], padding=3, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(REPS, fontsize=9)
        ax.set_ylim(0, 0.8)
        ax.set_ylabel(f"{clf_label}\n\nmacro-F1", fontsize=10, fontweight="bold")
        if row == 0:
            ax.set_title("Native pose\n(test = training orientation)", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)

        # -- Col 1: rotated test — train native vs train SO(3) (augmentation) --
        ax = axes[row, 1]
        rot_rr = [_agg(rows, classifier=clf, rep=r, train_mode="rot", test_mode="rot")[0] for r in REPS]
        b1 = ax.bar(x - 0.19, rot, 0.34, color=[_COLORS[r] for r in REPS])
        b2 = ax.bar(x + 0.19, rot_rr, 0.34, color=[_COLORS[r] for r in REPS],
                    hatch="////", edgecolor="white", linewidth=0)
        for xi, r, n in zip(x, REPS, nat):  # dotted = native-pose level (shows the drop)
            ax.hlines(n, xi - 0.37, xi + 0.37, color=_COLORS[r], ls=":", lw=1.4, alpha=0.9)
        ax.bar_label(b1, labels=[f"{v:.2f}" for v in rot], padding=2, fontsize=7)
        ax.bar_label(b2, labels=[f"{v:.2f}" for v in rot_rr], padding=2, fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(REPS, fontsize=9)
        ax.set_ylim(0, 0.8)
        if row == 0:
            ax.set_title("Accuracy on randomly-oriented (SO(3)) test objects\n"
                         "two training strategies vs the no-rotation baseline", fontsize=10.5)
            ax.legend(handles=[
                Patch(facecolor="0.55", label="train native pose → test rotated"),
                Patch(facecolor="0.55", hatch="////", edgecolor="white",
                      label="train rotated → test rotated (augmentation)"),
                Line2D([0], [0], color="0.5", ls=":", lw=1.4,
                       label="baseline: no rotation (train & test native)")],
                frameon=False, fontsize=7.3, loc="upper center", ncol=1, handlelength=1.8)
        ax.spines[["top", "right"]].set_visible(False)

        # -- Col 2: efficiency (feature size vs rotated score) --
        ax = axes[row, 2]
        dims = [feat_dim[r] for r in REPS]
        bars = ax.bar(x, dims, 0.62, color=[_COLORS[r] for r in REPS])
        ax.set_yscale("log")
        ax.bar_label(bars, labels=[f"{feat_dim[r]}-dim\nSO(3) F1={f:.2f}" for r, f in zip(REPS, rot)],
                     padding=3, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(REPS, fontsize=9)
        ax.set_ylim(10, max(dims) * 30)
        ax.set_ylabel("feature dimension (log)", fontsize=9)
        if row == 0:
            ax.set_title("Efficiency\n(feature size vs rotated score)", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "MedMNIST3D binary shape masks — where the DIWECT wins   (rows: SVM top, MLP bottom; "
        "mean over adrenal & vessel, macro-F1)\n"
        "Native pose favours the directional WECT; under SO(3) rotation only the invariant DIWECT "
        "holds its native level (dotted),\n"
        "and it does so with a 48-dim feature — while the flexible MLP overfits the training pose worst.",
        fontsize=10.5, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--svm-C", type=float, default=10.0)
    p.add_argument("--gray-percentile", type=float, default=55.0)
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    p.add_argument("--tune-wds", type=float, nargs="+", default=[1e-4])
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "medmnist3d_summary.csv"))
    p.add_argument("--datasets", nargs="+", default=None,
                   help="subset of datasets to run (default: all in DATASETS)")
    p.add_argument("--append", action="store_true",
                   help="append rows to an existing CSV instead of overwriting")
    p.add_argument("--figure-only", action="store_true")
    args = p.parse_args()
    args.out = os.path.abspath(args.out)
    fig_path = os.path.join(HERE, "medmnist3d_summary.png")

    if not args.figure_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        t0 = time.time()
        new_file = not (args.append and os.path.exists(args.out))
        with open(args.out, "a" if args.append else "w", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(_CSV)
            for name in (args.datasets or DATASETS):
                print(f"[{name}] ...", flush=True)
                run_dataset(name, args, w)
                f.flush()
        print(f"[summary written -> {args.out} in {time.time() - t0:.0f}s]")
    make_figure(args.out, fig_path)


if __name__ == "__main__":
    main()
