"""Real-world 3D analogue of ``real/real_classification.py`` on ModelNet10.

Ten classes of household CAD meshes, classified from five representations built
on top of a uniform voxel resampling of each object:

    voxels          raw occupancy grid (the raw-geometry baseline)
    wect            full Weighted Euler Characteristic Transform matrix
    diwect          direction-integrated WECT (a single height curve)
    voxels+wect     concatenation
    voxels+diwect   concatenation

We reproduce the four probes from the 2D real experiment -- head-to-head,
data-efficiency, capacity (accuracy vs head size), and rotation robustness.
ModelNet objects come in a canonical upright pose, so the directional WECT
starts with orientation as *signal*; the rotation panel then spins the test
objects about the vertical axis (exactly, on the mesh) and asks which
representation keeps its accuracy -- the regime where DIWECT's baked-in
direction averaging should pay off.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from core.wect import WECTTransform
from core.mlp import train_mlp, predict, tune_head
from core.meshes import mesh_to_complex, voxelize
from core.modelnet import load_modelnet10, load_normalized_mesh, resample_mesh

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
WORLD_HALF = 1.45  # voxel world box; fits any azimuthal rotation of a unit shape
REPRESENTATIONS = ["voxels", "wect", "diwect", "voxels+wect", "voxels+diwect"]
_COLORS = {"voxels": "#8d99ae", "wect": "#d1495b", "diwect": "#30638e",
           "voxels+wect": "#9b5de5", "voxels+diwect": "#f4a261"}
_DISPLAY = {"voxels+wect": "vox+WECT", "voxels+diwect": "vox+DIWECT"}


def _rot_axis(angle_deg: float, axis: int = 2) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(3, dtype=np.float32)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _feat_uniform(uni, T, grid):
    """Features of an already-uniform mesh: voxelize it, and read its WECT/DIWECT.

    Both the canonical pass and the rotation test call this, so an exact
    rotation of ``uni`` feeds through identically -- the WECT/DIWECT see a true
    rotation of the complex (DIWECT stays invariant) while the voxel grid is
    freshly rasterized (rotation-sensitive, as a raw baseline should be).
    """
    occ = voxelize(uni.verts, uni.faces, grid=grid, half=WORLD_HALF)
    cplx = mesh_to_complex(uni.verts, uni.faces)
    return occ.ravel().astype(np.float32), T.wect(cplx).ravel(), T.diwect(cplx)


# --------------------------------------------------------------------------- #
# Features (extract once, cache, reuse)
# --------------------------------------------------------------------------- #
def build_features(cfg):
    ds = load_modelnet10(CACHE_DIR, seed=cfg.split_seed)
    key = (f"feats_modelnet10_g{cfg.vox_grid}_d{cfg.num_dirs}_h{cfg.num_heights}"
           f"_s{cfg.split_seed}.npz")
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        z = np.load(path)
        print(f"[loaded cached features {key}]")
        base = {"voxels": z["voxels"], "wect": z["wect"], "diwect": z["diwect"]}
        return base, z["labels"], (z["tr"], z["va"], z["te"]), int(z["n_classes"])

    files, labels = ds["files"], ds["labels"]
    tr, va, te = ds["split"]
    n = len(files)
    print(f"[modelnet10: extracting features for {n} meshes ...]")
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights, dim=3)

    t0 = time.time()
    voxels = np.empty((n, cfg.vox_grid ** 3), np.float32)
    wect = np.empty((n, cfg.num_dirs * cfg.num_heights), np.float32)
    diwect = np.empty((n, cfg.num_heights), np.float32)
    for i, path_i in enumerate(files):
        uni = resample_mesh(load_normalized_mesh(path_i), cfg.vox_grid, half=WORLD_HALF)
        voxels[i], wect[i], diwect[i] = _feat_uniform(uni, T, cfg.vox_grid)
        if i % 500 == 0:
            print(f"    {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  featurized in {time.time() - t0:.0f}s")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, voxels=voxels, wect=wect, diwect=diwect, labels=labels,
                        tr=tr, va=va, te=te, n_classes=ds["n_classes"])
    print(f"[cached -> {key}]")
    return {"voxels": voxels, "wect": wect, "diwect": diwect}, labels, (tr, va, te), ds["n_classes"]


def assemble(base, rep, idx=None):
    def take(name):
        return base[name] if idx is None else base[name][idx]
    if rep in base:
        return take(rep)
    return np.concatenate([take(p) for p in rep.split("+")], axis=1)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def _splits(base, tr, va, te):
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


def _stratified_subsample(y, frac, rng):
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        k = max(1, int(round(frac * len(pool))))
        idx.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(idx)


def run_data_efficiency(feats, y, tr, va, te, cfg, hp):
    out = {rep: {"n": [], "mean": [], "std": []} for rep in REPRESENTATIONS}
    tmp = {rep: {f: [] for f in cfg.fracs} for rep in REPRESENTATIONS}
    for frac in cfg.fracs:
        for seed in range(cfg.seeds):
            rng = np.random.default_rng(1000 * seed + int(frac * 1000))
            sub = _stratified_subsample(y[tr], frac, rng)
            for rep in REPRESENTATIONS:
                Xtr, Xva, Xte = feats[rep]
                r = train_mlp(Xtr[sub], y[tr][sub], Xva, y[va], Xte, y[te],
                              width=cfg.head_width, epochs=cfg.epochs, seed=seed,
                              lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
                tmp[rep][frac].append(r["test_acc"])
    n_tr = len(tr)
    for rep in REPRESENTATIONS:
        for frac in cfg.fracs:
            out[rep]["n"].append(int(round(frac * n_tr)))
            out[rep]["mean"].append(float(np.mean(tmp[rep][frac])))
            out[rep]["std"].append(float(np.std(tmp[rep][frac])))
    return out


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


def run_rotation(base, y, tr, va, te, cfg, hp):
    heads = {}
    for rep in REPRESENTATIONS:
        Xtr, Xva, Xte = (assemble(base, rep, tr), assemble(base, rep, va),
                         assemble(base, rep, te))
        r = train_mlp(Xtr, y[tr], Xva, y[va], Xte, y[te],
                      width=cfg.head_width, epochs=cfg.epochs, seed=0,
                      lr=hp[rep]["lr"], weight_decay=hp[rep]["weight_decay"])
        heads[rep] = (r["model"], r["scaler"])

    ds = load_modelnet10(CACHE_DIR, seed=cfg.split_seed)
    rng = np.random.default_rng(0)
    sub = _stratified_subsample(y[te], min(1.0, cfg.rot_n / len(te)), rng)
    te_sub = np.sort(te[sub])
    y_sub = y[te_sub]
    # Resample each object to a uniform mesh once, then rotate *that* mesh
    # exactly per angle (no re-voxelization of the raw object -> no aliasing).
    unis = [resample_mesh(load_normalized_mesh(ds["files"][i]), cfg.vox_grid, half=WORLD_HALF)
            for i in te_sub]
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights, dim=3)

    out = {rep: {"angles": cfg.angles, "acc": []} for rep in REPRESENTATIONS}
    for ang in cfg.angles:
        R = _rot_axis(ang, axis=cfg.rot_axis)
        m = len(y_sub)
        vox = np.empty((m, cfg.vox_grid ** 3), np.float32)
        wect = np.empty((m, cfg.num_dirs * cfg.num_heights), np.float32)
        diw = np.empty((m, cfg.num_heights), np.float32)
        for j, uni in enumerate(unis):
            rot = uni._replace(verts=uni.verts @ R.T) if ang else uni
            vox[j], wect[j], diw[j] = _feat_uniform(rot, T, cfg.vox_grid)
        rot_base = {"voxels": vox, "wect": wect, "diwect": diw}
        for rep in REPRESENTATIONS:
            model, scaler = heads[rep]
            pred = predict(model, scaler, assemble(rot_base, rep))
            out[rep]["acc"].append(float(np.mean(pred == y_sub)))
        print(f"    rotation {ang:3d}deg done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Report + figure
# --------------------------------------------------------------------------- #
def _report(results):
    cfg = results["cfg"]
    h2h = results["head_to_head"]
    print(f"\nmodelnet10 | {results['n_classes']} classes | uniform {cfg.vox_grid}^3 voxels + "
          f"WECT/DIWECT | head width {cfg.head_width}, {cfg.seeds} seeds")
    print("head params = trainable params of the MLP head (the WECT transform adds none).")
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
    vd, vw = h2h["voxels+diwect"]["acc"].mean(), h2h["voxels+wect"]["acc"].mean()
    vx = h2h["voxels"]["acc"].mean()
    print(f"DIWECT vs WECT: {d:.3f} vs {w:.3f} ({d - w:+.3f}).  "
          f"voxels {vx:.3f} -> vox+DIWECT {vd:.3f} ({vd - vx:+.3f}), "
          f"vox+WECT {vw:.3f} ({vw - vx:+.3f}).")
    rot = results["rotation"]
    worst = lambda r: float(np.min(rot[r]["acc"]))
    print(f"Rotation about axis {cfg.rot_axis} (worst-angle test acc; higher = more stable): "
          f"DIWECT {worst('diwect'):.3f} vs WECT {worst('wect'):.3f}; "
          f"vox+DIWECT {worst('voxels+diwect'):.3f} vs vox+WECT {worst('voxels+wect'):.3f}.")
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
    ax.set_xscale("log"); ax.set_xlabel("head parameters (WECT not counted)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Capacity: accuracy vs head size", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # (4) Rotation robustness
    ax = axes[1, 1]
    for rep in reps:
        r = results["rotation"][rep]
        ax.plot(r["angles"], r["acc"], color=_COLORS[rep], lw=2, marker="o", ms=4,
                label=_DISPLAY.get(rep, rep))
    ax.set_xlabel(f"test-set rotation about axis {cfg.rot_axis} (degrees)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Rotation robustness of the test set (exact mesh rotation)", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ModelNet10: canonical pose favours the directional WECT, but the DIWECT "
                 "keeps its accuracy when objects are rotated", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(cfg):
    base, labels, (tr, va, te), n_classes = build_features(cfg)
    if cfg.max_train and len(tr) > cfg.max_train:
        tr = np.random.default_rng(cfg.split_seed).permutation(tr)[:cfg.max_train]
    print(f"[split: {len(tr)} train / {len(va)} val / {len(te)} test | {n_classes} classes]")
    feats = _splits(base, tr, va, te)

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
    p.add_argument("--vox-grid", type=int, default=28)
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--head-width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--cap-epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--angles", type=int, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--rot-axis", type=int, default=2, help="axis to rotate about (0=x,1=y,2=z)")
    p.add_argument("--rot-n", type=int, default=908)
    p.add_argument("--thr-frac", type=float, default=0.90)
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0, help="cap train size (0 = all)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "modelnet10_classification.png")

    if args.quick:
        args.seeds = 1
        args.epochs, args.cap_epochs = 40, 30
        args.widths = [4, 16, 64]
        args.fracs = [0.05, 0.25, 1.0]
        args.angles = [0, 90, 180, 270]
        args.rot_n = 300
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]

    results = run(args)
    make_figure(results, args.out)


if __name__ == "__main__":
    main()
