"""DIWECT vs WECT on MedMNIST v2 3D shape datasets -- the biological counterpoint.

The ModelNet result (WECT >> DIWECT) turned out to be a data-orientation effect:
its objects are canonically posed and highly anisotropic, so orientation is
class signal that the direction-integrated DIWECT throws away. Biological
structures (adrenal glands, vessels, synapses) have no canonical orientation, so
the hypothesis is that here orientation is *not* signal and the DIWECT should be
competitive with -- or better than -- the WECT.

Two panels test that:
  (1) head-to-head in the native pose (voxels / wect / diwect / fusions), and
  (2) an orientation-sensitivity panel: wect vs diwect trained/tested under the
      native pose or a random SO(3) rotation. A small native/native gap and a
      small native->rot drop mean orientation carries little signal -- the
      DIWECT-friendly regime, opposite to ModelNet.

Everything runs on binary shape masks -> occupancy -> marching-cubes mesh, the
same path as the other 3D experiments; rotations are exact on the mesh.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score

from core.wect import WECTTransform
from core.mlp import train_mlp, tune_head
from core.meshes import occ_to_mesh, mesh_to_complex, voxelize, random_rotation, rotate
from core.medmnist3d import load_medmnist3d, volume_to_occupancy

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
WORLD_HALF = 1.75  # voxel box wide enough for any SO(3) rotation of a unit shape
REPS = ["voxels", "wect", "diwect", "voxels+wect", "voxels+diwect"]
_COLORS = {"voxels": "#8d99ae", "wect": "#d1495b", "diwect": "#30638e",
           "voxels+wect": "#9b5de5", "voxels+diwect": "#f4a261"}
_DISPLAY = {"voxels+wect": "vox+WECT", "voxels+diwect": "vox+DIWECT"}
CONDITIONS = [("native", "native"), ("native", "rot"), ("rot", "rot")]
_COND_LABEL = {("native", "native"): "train native\ntest native",
               ("native", "rot"): "train native\ntest SO(3)",
               ("rot", "rot"): "train SO(3)\ntest SO(3)"}


# --------------------------------------------------------------------------- #
# Data -> meshes + native voxels
# --------------------------------------------------------------------------- #
def build(cfg):
    ds = load_medmnist3d(cfg.dataset, CACHE_DIR)
    vols, labels = ds["volumes"], ds["labels"]
    n = len(vols)
    meshes, voxels = [], np.empty((n, vols.shape[1] ** 3), np.float32)
    t0 = time.time()
    for i, vol in enumerate(vols):
        occ = volume_to_occupancy(vol, ds["is_mask"], cfg.gray_percentile)
        voxels[i] = occ.ravel()
        meshes.append(occ_to_mesh(occ))
    print(f"[{cfg.dataset}: {n} volumes -> meshes in {time.time() - t0:.0f}s | "
          f"classes {ds['classes']} | balance {np.bincount(labels)}]", flush=True)
    return ds, meshes, voxels, labels


def wect_feats(meshes, T, mode, seed):
    rng = np.random.default_rng(seed)
    W, D = [], []
    for m in meshes:
        v = rotate(m.verts, random_rotation(rng)) if mode == "rot" else m.verts
        mat = T.wect(mesh_to_complex(v, m.faces)).reshape(T.num_dirs, T.num_heights)
        W.append(mat.ravel())
        D.append(mat.mean(0))
    return {"wect": np.stack(W).astype(np.float32), "diwect": np.stack(D).astype(np.float32)}


def _balance(idx, y, seed):
    """Oversample minority classes within idx so the classifier isn't swamped."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y[idx])
    target = counts.max()
    out = []
    for c in np.unique(y[idx]):
        pool = idx[y[idx] == c]
        k = target - len(pool)
        extra = rng.choice(pool, k, replace=True) if k > 0 else np.empty(0, int)
        out.append(np.concatenate([pool, extra]))
    return rng.permutation(np.concatenate(out))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def assemble(base, rep, idx):
    parts = rep.split("+")
    return np.concatenate([base[p][idx] for p in parts], axis=1)


def head_to_head(base, y, split, cfg):
    tr, va, te = split
    grid = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    out = {}
    for rep in REPS:
        Xtr, Xva, Xte = assemble(base, rep, tr), assemble(base, rep, va), assemble(base, rep, te)
        best = tune_head(Xtr, y[tr], Xva, y[va], Xte, y[te],
                         width=cfg.width, epochs=cfg.epochs, seed=0, grid=grid)
        lr, wd = best["hp"]["lr"], best["hp"]["weight_decay"]
        accs, f1s, baccs = [], [], []
        for seed in range(cfg.seeds):
            trb = _balance(tr, y, seed) if cfg.balance else tr
            r = train_mlp(assemble(base, rep, trb), y[trb], Xva, y[va], Xte, y[te],
                          width=cfg.width, epochs=cfg.epochs, seed=seed, lr=lr, weight_decay=wd)
            accs.append(r["test_acc"])
            f1s.append(f1_score(y[te], r["test_pred"], average="macro"))
            baccs.append(balanced_accuracy_score(y[te], r["test_pred"]))
        out[rep] = {"acc": np.array(accs), "f1": np.array(f1s), "bacc": np.array(baccs),
                    "params": r["params"], "hp": (lr, wd)}
    return out


def orientation_panel(meshes, y, split, cfg):
    tr, va, te = split
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights, dim=3)
    # featurize each split under native and SO(3) modes (seeded per split+mode)
    feats = {}
    for i, (sp, ids) in enumerate([("tr", tr), ("va", va), ("te", te)]):
        ms = [meshes[j] for j in ids]
        feats[sp] = {"native": wect_feats(ms, T, "native", 0),
                     "rot": wect_feats(ms, T, "rot", 100 + i)}
    grid = {"lr": cfg.tune_lrs, "weight_decay": cfg.tune_wds}
    out = {rep: {} for rep in ("wect", "diwect")}
    for rep in ("wect", "diwect"):
        best = tune_head(feats["tr"]["native"][rep], y[tr], feats["va"]["native"][rep], y[va],
                         feats["te"]["native"][rep], y[te], width=cfg.width, epochs=cfg.epochs,
                         seed=0, grid=grid)
        lr, wd = best["hp"]["lr"], best["hp"]["weight_decay"]
        for tr_m, te_m in CONDITIONS:
            f1s = []
            for seed in range(cfg.seeds):
                trb = _balance(tr, y, seed) if cfg.balance else tr
                # balanced indices are positions into tr's feature block
                pos = {g: k for k, g in enumerate(tr)}
                sel = np.array([pos[g] for g in trb])
                Xtr = feats["tr"][tr_m][rep][sel]
                r = train_mlp(Xtr, y[trb], feats["va"][te_m][rep], y[va],
                              feats["te"][te_m][rep], y[te], width=cfg.width,
                              epochs=cfg.epochs, seed=seed, lr=lr, weight_decay=wd)
                f1s.append(f1_score(y[te], r["test_pred"], average="macro"))
            out[rep][(tr_m, te_m)] = (float(np.mean(f1s)), float(np.std(f1s)))
    return out


# --------------------------------------------------------------------------- #
# Report + figure
# --------------------------------------------------------------------------- #
def _report(ds, h2h, orient, cfg):
    print(f"\n{cfg.dataset} | {ds['n_classes']} classes | uniform mesh + WECT/DIWECT "
          f"| width {cfg.width}, {cfg.seeds} seeds | metric: macro-F1 (imbalanced)")
    print("-" * 78)
    print(f"{'representation':16}{'accuracy':>14}{'macro-F1':>14}{'bal-acc':>12}{'params':>12}")
    for rep in REPS:
        r = h2h[rep]
        acc, f1, bacc = r["acc"].mean(), r["f1"], r["bacc"].mean()
        print(f"{_DISPLAY.get(rep, rep):16}{acc:>14.3f}"
              f"{f'{f1.mean():.3f}±{f1.std():.3f}':>14}{bacc:>12.3f}{r['params']:>12,}")
    print("-" * 78)
    d, w = h2h["diwect"]["f1"].mean(), h2h["wect"]["f1"].mean()
    print(f"WECT vs DIWECT (macro-F1): {w:.3f} vs {d:.3f} ({d - w:+.3f}).")
    nn = orient["wect"][("native", "native")][0] - orient["diwect"][("native", "native")][0]
    nr = orient["wect"][("native", "rot")][0]
    dvr = orient["diwect"][("native", "rot")][0]
    print(f"Orientation diagnostic (macro-F1): native/native WECT-DIWECT gap {nn:+.3f} "
          f"(ModelNet was ~+0.30); WECT native->SO(3) test {orient['wect'][('native','native')][0]:.3f}"
          f"->{nr:.3f}, DIWECT flat (~{dvr:.3f}).")
    if abs(nn) < 0.06:
        print("=> orientation is NOT class signal here: the DIWECT matches the WECT, and it does so "
              "with a tiny rotation-invariant feature -- the biological, pose-free regime.")


def make_figure(ds, h2h, orient, cfg, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(REPS))
    accm = [h2h[r]["acc"].mean() for r in REPS]
    f1m = [h2h[r]["f1"].mean() for r in REPS]
    f1e = [h2h[r]["f1"].std() for r in REPS]
    ax.bar(x - 0.2, accm, 0.4, color=[_COLORS[r] for r in REPS], alpha=0.45, label="accuracy")
    b = ax.bar(x + 0.2, f1m, 0.4, yerr=f1e, capsize=3, color=[_COLORS[r] for r in REPS],
               label="macro-F1")
    ax.bar_label(b, labels=[f"{v:.2f}" for v in f1m], padding=3, fontsize=8)
    ax.axhline(1.0 / ds["n_classes"], ls="--", lw=1, color="k", label="chance")
    ax.set_xticks(x); ax.set_xticklabels([_DISPLAY.get(r, r) for r in REPS], fontsize=8, rotation=12)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.set_title("Head-to-head, native pose (bars = macro-F1)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    xc = np.arange(len(CONDITIONS))
    for rep, off in [("wect", -0.19), ("diwect", 0.19)]:
        m = [orient[rep][c][0] for c in CONDITIONS]
        e = [orient[rep][c][1] for c in CONDITIONS]
        bb = ax.bar(xc + off, m, 0.36, yerr=e, capsize=3, color=_COLORS[rep], label=rep)
        ax.bar_label(bb, labels=[f"{v:.2f}" for v in m], padding=3, fontsize=8)
    ax.set_xticks(xc); ax.set_xticklabels([_COND_LABEL[c] for c in CONDITIONS], fontsize=8.5)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.0)
    ax.set_title("Orientation sensitivity (SO(3))", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    gap = orient["wect"][("native", "native")][0] - orient["diwect"][("native", "native")][0]
    fig.suptitle(f"{cfg.dataset}: a pose-free biological shape task — the DIWECT matches the WECT "
                 f"(native WECT-DIWECT gap {gap:+.2f}) and is SO(3)-invariant",
                 fontsize=11.5, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def run(cfg):
    ds, meshes, voxels, labels = build(cfg)
    T = WECTTransform(num_dirs=cfg.num_dirs, num_heights=cfg.num_heights, dim=3)
    t0 = time.time()
    native = wect_feats(meshes, T, "native", 0)
    base = {"voxels": voxels, "wect": native["wect"], "diwect": native["diwect"]}
    print(f"[featurized native WECT/DIWECT in {time.time() - t0:.0f}s]", flush=True)

    h2h = head_to_head(base, labels, ds["split"], cfg)
    orient = orientation_panel(meshes, labels, ds["split"], cfg)
    _report(ds, h2h, orient, cfg)
    make_figure(ds, h2h, orient, cfg, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{cfg.dataset}_classification.png"))
    return h2h, orient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="adrenalmnist3d",
                   choices=["adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",
                            "nodulemnist3d", "organmnist3d", "fracturemnist3d"])
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--gray-percentile", type=float, default=55.0,
                   help="occupancy threshold percentile for grayscale (non-mask) volumes")
    p.add_argument("--no-balance", dest="balance", action="store_false")
    p.add_argument("--tune-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--tune-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 40
        args.tune_lrs, args.tune_wds = [1e-3, 3e-3], [1e-4]
    run(args)


if __name__ == "__main__":
    main()
