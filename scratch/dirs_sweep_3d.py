"""Direction (num_dirs) sweep for the 3D mesh experiments, for two classifiers.

Answers "how do the 3D conclusions depend on the number of directions, and does
a weaker model change the picture?" by sweeping ``num_dirs`` up to ~256 for both

    * a distance-kernel RBF **SVM** -- a fixed-similarity model that *cannot*
      learn to undo a rotation, and
    * a one-hidden-layer **MLP** -- which can, given width and data,

over the representations ``wect`` / ``wect_aligned`` (oracle) / ``diwect``, with
the raw ``voxels`` occupancy as a direction-independent reference.

Two things matter for the sphere. First, DIWECT's feature dimension is always
``num_heights`` regardless of ``num_dirs`` (it averages the directions away), so
more directions only sharpen its Monte-Carlo estimate of the direction integral
-- they never enlarge its head. Second, we draw a *fresh* Fibonacci cover of S^2
at every ``num_dirs`` (rather than subsampling one big set), because a
subsampled Fibonacci spiral is not a uniform cover and would bias the average.

Datasets: ``synthetic`` (volume-matched solids, mirrors mesh_classification) and
``modelnet10`` (mirrors mesh_real_classification).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from core.wect import WECTTransform
from core.mlp import train_mlp, tune_head
from core.meshes import make_mesh_library, mesh_to_complex, voxelize
from synthetic.classification3d import mesh_classification as mc3
from numdirs_sweep import CsvWriter

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = ["wect", "wect_aligned", "diwect"]
_ORACLE = {"wect_aligned"}


# --------------------------------------------------------------------------- #
# Distance-kernel RBF SVM (the "simpler model": fixed similarity, no learning
# of rotation). Median-heuristic gamma, same recipe as the 2D svm experiment.
# --------------------------------------------------------------------------- #
def svm_predict(Xtr, ytr, Xte, C=10.0):
    d2_tr = cdist(Xtr, Xtr, "sqeuclidean")
    gamma = 1.0 / (np.median(d2_tr[d2_tr > 0]) + 1e-12)
    clf = SVC(kernel="precomputed", C=C).fit(np.exp(-gamma * d2_tr), ytr)
    d2_te = cdist(Xte, Xtr, "sqeuclidean")
    return clf.predict(np.exp(-gamma * d2_te))


# --------------------------------------------------------------------------- #
# Features: WECT (+ its direction-mean DIWECT) from a list of complexes at a
# given num_dirs. One WECT computation yields both.
# --------------------------------------------------------------------------- #
def wect_feats(complexes, d, H):
    T = WECTTransform(num_dirs=d, num_heights=H, dim=3)
    mats = [T.wect(c).reshape(d, H) for c in complexes]
    wect = np.stack([m.ravel() for m in mats]).astype(np.float32)
    diwect = np.stack([m.mean(0) for m in mats]).astype(np.float32)
    return wect, diwect


def rep_feats(pool_split, d, H, reps):
    """Return {rep: X} for one split, computing only the representations needed."""
    crot, ccan, _vox, _y = pool_split
    out = {}
    if "wect" in reps or "diwect" in reps:
        w_rot, diw = wect_feats(crot, d, H)
        if "wect" in reps:
            out["wect"] = w_rot
        if "diwect" in reps:
            out["diwect"] = diw
    if "wect_aligned" in reps:  # oracle: only the synthetic arm uses a separate pose
        out["wect_aligned"], _ = wect_feats(ccan, d, H)
    return out


# --------------------------------------------------------------------------- #
# Pools of complexes (built once; re-featurized per num_dirs)
# --------------------------------------------------------------------------- #
def build_synth_pool(cfg, seed):
    lib = make_mesh_library(grid=cfg.gen_grid, volume=cfg.volume)
    out = {}
    for nm, npc, sd in [("tr", cfg.n_train, 10 * seed + 1),
                        ("va", cfg.n_val, 10 * seed + 2),
                        ("te", cfg.n_test, 10 * seed + 3)]:
        samples, y = mc3.make_dataset(lib, npc, sd)
        crot = [mesh_to_complex(rot, faces) for _c, rot, faces in samples]
        ccan = [mesh_to_complex(canon, faces) for canon, _r, faces in samples]
        vox = np.stack([voxelize(rot, faces, grid=cfg.vox_grid).ravel()
                        for _c, rot, faces in samples]).astype(np.float32)
        out[nm] = (crot, ccan, vox, y)
    return out


def build_modelnet_pool(cfg, seed):
    from core.modelnet import load_modelnet10, load_normalized_mesh, resample_mesh
    from real3d.mesh_real_classification import WORLD_HALF
    ds = load_modelnet10(os.path.join(HERE, "real3d", "cache"), seed=cfg.split_seed)
    tr, va, te = ds["split"]
    y = ds["labels"]
    # Cap sizes for the O(n^2)-kernel SVM to stay tractable.
    rng = np.random.default_rng(seed)
    def cap(idx, n):
        return idx if (n <= 0 or len(idx) <= n) else np.sort(rng.permutation(idx)[:n])
    idx = {"tr": cap(tr, cfg.mn_max_train), "va": cap(va, cfg.mn_max_val),
           "te": cap(te, cfg.mn_max_test)}
    out = {}
    for nm, ids in idx.items():
        crot, ccan, vox = [], [], []
        for i in ids:
            uni = resample_mesh(load_normalized_mesh(ds["files"][i]), cfg.vox_grid, half=WORLD_HALF)
            crot.append(mesh_to_complex(uni.verts, uni.faces))
            ccan.append(uni)  # canonical == uniform mesh; no separate "aligned" pose
            vox.append(voxelize(uni.verts, uni.faces, grid=cfg.vox_grid, half=WORLD_HALF).ravel())
        # ModelNet is already in canonical pose, so "wect_aligned" == "wect".
        ccan = [mesh_to_complex(m.verts, m.faces) for m in ccan]
        out[nm] = (crot, ccan, np.stack(vox).astype(np.float32), y[ids])
    return out


MEDMNIST3D = {"adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",
              "nodulemnist3d", "organmnist3d", "fracturemnist3d"}


def _balance_idx(idx, y, seed):
    """Oversample minority classes within idx (MedMNIST sets are imbalanced)."""
    rng = np.random.default_rng(seed)
    target = np.bincount(y[idx]).max()
    out = []
    for c in np.unique(y[idx]):
        pool = idx[y[idx] == c]
        extra = rng.choice(pool, target - len(pool), replace=True) if len(pool) < target \
            else np.empty(0, int)
        out.append(np.concatenate([pool, extra]))
    return rng.permutation(np.concatenate(out))


def build_medmnist_pool(cfg, seed):
    from core.medmnist3d import load_medmnist3d, volume_to_occupancy
    from core.meshes import occ_to_mesh
    ds = load_medmnist3d(cfg.dataset, os.path.join(HERE, "real3d", "cache"))
    vols, y = ds["volumes"], ds["labels"]
    tr, va, te = ds["split"]
    # native pose (no baked-in rotation); oversample the train split for balance
    idx = {"tr": _balance_idx(tr, y, seed) if cfg.balance else tr, "va": va, "te": te}
    # build each unique mesh once, then index (train oversampling reuses complexes)
    cache = {}
    def complex_of(i):
        if i not in cache:
            occ = volume_to_occupancy(vols[i], ds["is_mask"], cfg.gray_percentile)
            m = occ_to_mesh(occ)
            cache[i] = (mesh_to_complex(m.verts, m.faces), occ.ravel().astype(np.float32))
        return cache[i]
    out = {}
    for nm, ids in idx.items():
        cx = [complex_of(i) for i in ids]
        crot = [c for c, _v in cx]
        vox = np.stack([v for _c, v in cx]).astype(np.float32)
        out[nm] = (crot, crot, vox, y[ids])  # native == "aligned"; ccan unused (reps skip it)
    return out


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def _dh(cfg, val):
    """Map a swept grid value to (num_dirs, num_heights), holding the other axis."""
    return (val, cfg.num_heights) if cfg.sweep == "dirs" else (cfg.num_dirs, val)


def run(cfg):
    writer = CsvWriter(cfg.out, resume=cfg.resume)
    build = {"synthetic": build_synth_pool, "modelnet10": build_modelnet_pool}.get(
        cfg.dataset, build_medmnist_pool)
    grid_hp = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    reps = REPS if cfg.dataset == "synthetic" else ["wect", "diwect"]

    t0 = time.time()
    pools = {seed: build(cfg, seed) for seed in range(cfg.seeds)}
    print(f"[{cfg.dataset}: sweep={cfg.sweep} | built {cfg.seeds} pools in {time.time() - t0:.0f}s "
          f"| {len(pools[0]['tr'][3])} train / {len(pools[0]['te'][3])} test]", flush=True)

    # (0) axis-independent voxel reference (both grid columns marked 0)
    for seed in range(cfg.seeds):
        _, _, vtr, ytr = pools[seed]["tr"]
        _, _, vte, yte = pools[seed]["te"]
        fdim = vtr.shape[1]
        if cfg.do_svm:
            pred = svm_predict(vtr, ytr, vte, C=cfg.svm_C)
            writer.row(cfg.dataset, cfg.dataset, "svm", "voxels", 0, 0, fdim, "",
                       seed, accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
        if cfg.do_mlp:
            _, _, vva, yva = pools[seed]["va"]
            lr, wd = _tune(vtr, ytr, vva, yva, vte, yte, cfg, grid_hp) if seed == 0 else (lr, wd)
            r = train_mlp(vtr, ytr, vva, yva, vte, yte, width=cfg.mlp_width,
                          epochs=cfg.mlp_epochs, seed=seed, lr=lr, weight_decay=wd)
            writer.row(cfg.dataset, cfg.dataset, "mlp", "voxels", 0, 0, fdim, r["params"],
                       seed, r["test_acc"], f1_score(yte, r["test_pred"], average="macro"))

    # (1) MLP: tune lr/wd once per rep at the finest grid value (seed 0), hold across sweep
    mlp_hp = {}
    if cfg.do_mlp:
        dmax, Hmax = _dh(cfg, cfg.grid[-1])
        f0 = {sp: rep_feats(pools[0][sp], dmax, Hmax, reps) for sp in ("tr", "va", "te")}
        for rep in reps:
            ytr, yva, yte = (pools[0]["tr"][3], pools[0]["va"][3], pools[0]["te"][3])
            mlp_hp[rep] = _tune(f0["tr"][rep], ytr, f0["va"][rep], yva,
                                f0["te"][rep], yte, cfg, grid_hp)
            print(f"    tuned MLP {rep:14} lr={mlp_hp[rep][0]:.0e} wd={mlp_hp[rep][1]:.0e}", flush=True)

    # (2) sweep the chosen axis
    for val in cfg.grid:
        td = time.time()
        d, H = _dh(cfg, val)
        for seed in range(cfg.seeds):
            feats = {sp: rep_feats(pools[seed][sp], d, H, reps) for sp in ("tr", "va", "te")}
            ytr, yva, yte = (pools[seed]["tr"][3], pools[seed]["va"][3], pools[seed]["te"][3])
            for rep in reps:
                Xtr, Xva, Xte = feats["tr"][rep], feats["va"][rep], feats["te"][rep]
                fdim = Xtr.shape[1]
                if cfg.do_svm:
                    pred = svm_predict(Xtr, ytr, Xte, C=cfg.svm_C)
                    writer.row(cfg.dataset, cfg.dataset, "svm", rep, d, H, fdim, "",
                               seed, accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
                if cfg.do_mlp:
                    lr, wd = mlp_hp[rep]
                    r = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, width=cfg.mlp_width,
                                  epochs=cfg.mlp_epochs, seed=seed, lr=lr, weight_decay=wd)
                    writer.row(cfg.dataset, cfg.dataset, "mlp", rep, d, H, fdim, r["params"],
                               seed, r["test_acc"], f1_score(yte, r["test_pred"], average="macro"))
        print(f"  {cfg.sweep}={val:4d} done ({time.time() - td:.0f}s)", flush=True)
    writer.close()
    print(f"[sweep written -> {cfg.out} in {time.time() - t0:.0f}s]")


def _tune(Xtr, ytr, Xva, yva, Xte, yte, cfg, grid):
    best = tune_head(Xtr, ytr, Xva, yva, Xte, yte, width=cfg.mlp_width,
                     epochs=cfg.mlp_epochs, seed=0, grid=grid)
    return best["hp"]["lr"], best["hp"]["weight_decay"]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
_COLORS = {"wect": "#d1495b", "wect_aligned": "#2a9d8f", "diwect": "#30638e",
           "voxels": "#8d99ae"}
_DISPLAY = {"wect_aligned": "wect+align"}


def make_figure(csv_path, out_path, dataset):
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        print(f"[no rows for {dataset} in {csv_path}; skipping figure]")
        return
    clfs = [c for c in ["svm", "mlp"] if any(r["classifier"] == c for r in rows)]
    reps = [rp for rp in REPS if any(r["representation"] == rp for r in rows)]

    # Auto-detect the swept axis: the grid column that varies across non-voxel rows.
    nonvox = [r for r in rows if r["representation"] != "voxels"]
    axis = "num_heights" if len({r["num_heights"] for r in nonvox}) > 1 else "num_dirs"
    xlabel = {"num_dirs": "number of directions (num_dirs)",
              "num_heights": "number of filtration steps (num_heights)"}[axis]
    # Imbalanced MedMNIST sets: judge on macro-F1, not accuracy.
    metric = "macro_f1" if dataset in MEDMNIST3D else "accuracy"
    ylabel = f"test {'macro-F1' if metric == 'macro_f1' else 'accuracy'}"

    def curve(clf, rep):
        pts = {}
        for r in rows:
            if r["classifier"] == clf and r["representation"] == rep:
                pts.setdefault(int(r[axis]), []).append(float(r[metric]))
        ds = sorted(pts)
        return ds, np.array([np.mean(pts[d]) for d in ds]), np.array([np.std(pts[d]) for d in ds])

    def vox_level(clf):
        v = [float(r[metric]) for r in rows
             if r["classifier"] == clf and r["representation"] == "voxels"]
        return float(np.mean(v)) if v else None

    fig, axes = plt.subplots(1, len(clfs), figsize=(6.2 * len(clfs), 5.2),
                             squeeze=False, constrained_layout=True)
    for k, clf in enumerate(clfs):
        ax = axes[0, k]
        for rep in reps:
            ds, m, s = curve(clf, rep)
            if not ds:
                continue
            ax.errorbar(ds, m, yerr=s, color=_COLORS[rep], lw=2, marker="o", ms=5,
                        capsize=3, label=_DISPLAY.get(rep, rep),
                        ls="--" if rep in _ORACLE else "-")
        lvl = vox_level(clf)
        if lvl is not None:
            ax.axhline(lvl, color=_COLORS["voxels"], lw=1.5, ls=":",
                       label=f"voxels ({lvl:.2f})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({int(r[axis]) for r in nonvox if int(r[axis]) > 0}))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title({"svm": "Distance-kernel SVM (fixed similarity)",
                      "mlp": "MLP head (can learn rotation)"}[clf], fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    suptitle = {
        ("synthetic", "num_dirs"): (
            "Synthetic meshes, random 3D pose (few-shot): rotation is pure nuisance\n"
            "Neither model reads the raw WECT; the pose-free DIWECT hits the aligned oracle with a "
            "fixed 48-dim head once S² is covered (~32 dirs)"),
        ("modelnet10", "num_dirs"): (
            "ModelNet10, canonical pose: orientation is signal, not nuisance\n"
            "The directional WECT and raw voxels win; more directions help the WECT but make the "
            "rotation-invariant DIWECT worse"),
        ("synthetic", "num_heights"): (
            "Synthetic meshes: filtration-resolution sweep\n"
            "Accuracy saturates once num_heights resolves the mesh; past that, extra steps neither "
            "help nor hurt"),
        ("modelnet10", "num_heights"): (
            "ModelNet10: filtration-resolution sweep\n"
            "Where does accuracy stop improving as the height axis is refined?"),
    }.get((dataset, axis), (
        f"{dataset} ({'direction' if axis == 'num_dirs' else 'filtration-resolution'} sweep): "
        "pose-free biological shapes\n"
        "is the DIWECT competitive with the WECT across the sweep, and where does it saturate?"))
    fig.suptitle(suptitle, fontsize=10.5, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "modelnet10", "adrenalmnist3d", "vesselmnist3d",
                            "synapsemnist3d", "nodulemnist3d", "organmnist3d", "fracturemnist3d"])
    p.add_argument("--sweep", choices=["dirs", "heights"], default="dirs",
                   help="which axis to sweep; the other is held fixed")
    p.add_argument("--grid", type=int, nargs="+", default=None,
                   help="swept values (default: 2..256 for dirs, 6..192 for heights)")
    p.add_argument("--num-dirs", type=int, default=64, help="fixed num_dirs when --sweep heights")
    p.add_argument("--num-heights", type=int, default=48,
                   help="fixed num_heights when --sweep dirs")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-svm", dest="do_svm", action="store_false")
    p.add_argument("--no-mlp", dest="do_mlp", action="store_false")
    p.add_argument("--svm-C", type=float, default=10.0)
    p.add_argument("--mlp-width", type=int, default=64)
    p.add_argument("--mlp-epochs", type=int, default=120)
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    # synthetic shape settings
    p.add_argument("--gen-grid", type=int, default=32)
    p.add_argument("--volume", type=int, default=3200)
    p.add_argument("--vox-grid", type=int, default=24)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--n-test", type=int, default=100)
    # modelnet settings
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--mn-max-train", type=int, default=1200)
    p.add_argument("--mn-max-val", type=int, default=400)
    p.add_argument("--mn-max-test", type=int, default=908)
    # medmnist3d settings
    p.add_argument("--no-balance", dest="balance", action="store_false",
                   help="skip minority oversampling of the MedMNIST train split")
    p.add_argument("--gray-percentile", type=float, default=55.0,
                   help="occupancy threshold percentile for grayscale MedMNIST volumes")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--figure-only", action="store_true")
    args = p.parse_args()
    if args.dataset == "modelnet10":
        args.vox_grid = 28  # match the real experiment's grid
    if args.grid is None:
        args.grid = [2, 4, 8, 16, 32, 64, 128, 256] if args.sweep == "dirs" \
            else [6, 12, 24, 48, 96, 192]
    if args.out is None:
        stem = "dirs_sweep_3d" if args.sweep == "dirs" else "heights_sweep_3d"
        args.out = os.path.join(HERE, "results", f"{stem}_{args.dataset}.csv")

    if args.quick:
        args.grid = [4, 32, 256] if args.sweep == "dirs" else [12, 48, 192]
        args.seeds = 1
        args.n_train, args.n_val, args.n_test = 40, 30, 40
        args.mlp_epochs = 40
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]
        args.mn_max_train, args.mn_max_val, args.mn_max_test = 300, 150, 300

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig_path = args.out.replace(".csv", ".png")
    if not args.figure_only:
        run(args)
    make_figure(args.out, fig_path, args.dataset)


if __name__ == "__main__":
    main()
