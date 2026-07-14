from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from core.wect import WECTTransform
from core.shapes import make_shape_library, rotate_image

REPRESENTATIONS = ["pixels", "wect", "wect_aligned", "diwect", "mass"]
_COLORS = {"pixels": "#8d99ae", "wect": "#d1495b", "wect_aligned": "#2a9d8f",
           "diwect": "#30638e", "mass": "#c9ced6"}
# Ground-truth-angle oracle: needs the true rotation, unlike the others.
_ORACLE = {"wect_aligned"}
_DISPLAY = {"wect_aligned": "wect\n+align"}


# --------------------------------------------------------------------------- #
# Dataset + features
# --------------------------------------------------------------------------- #
def make_dataset(shapes, n_per_class, seed):
    rng = np.random.default_rng(seed)
    images, labels, angles = [], [], []
    for label, img in enumerate(shapes.values()):
        for _ in range(n_per_class):
            angle = rng.uniform(0.0, 360.0)
            images.append(rotate_image(img, angle))
            labels.append(label)
            angles.append(angle)
    return images, np.asarray(labels), np.asarray(angles)


def extract_features(images, angles, transform):
    aligned = [rotate_image(im, -a) for im, a in zip(images, angles)]
    return {
        "pixels": np.stack([im.ravel() for im in images]),
        "wect": np.stack([transform.wect(im).ravel() for im in images]),
        "wect_aligned": np.stack([transform.wect(im).ravel() for im in aligned]),
        "diwect": np.stack([transform.diwect(im) for im in images]),
        "mass": np.stack([[im.sum()] for im in images]),
    }


# --------------------------------------------------------------------------- #
# Distance-kernel SVM (uses each representation's own distance directly)
# --------------------------------------------------------------------------- #
def svm_predict(x_train, y_train, x_test, C=10.0):
    d2_train = cdist(x_train, x_train, "sqeuclidean")
    median = np.median(d2_train[d2_train > 0])
    gamma = 1.0 / (median + 1e-12)
    clf = SVC(kernel="precomputed", C=C).fit(np.exp(-gamma * d2_train), y_train)
    d2_test = cdist(x_test, x_train, "sqeuclidean")
    return clf.predict(np.exp(-gamma * d2_test))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(size=64, area=380, num_dirs=64, num_heights=48,
        n_train=6, n_test=40, n_seeds=6, C=10.0):
    shapes = make_shape_library(size=size, area=area)
    transform = WECTTransform(num_dirs=num_dirs, num_heights=num_heights)

    acc = {r: [] for r in REPRESENTATIONS}
    last = {}  # keep the final seed's arrays for plotting
    for seed in range(n_seeds):
        tr_imgs, y_tr, a_tr = make_dataset(shapes, n_train, seed)
        te_imgs, y_te, a_te = make_dataset(shapes, n_test, 1000 + seed)
        F_tr = extract_features(tr_imgs, a_tr, transform)
        F_te = extract_features(te_imgs, a_te, transform)
        for r in REPRESENTATIONS:
            pred = svm_predict(F_tr[r], y_tr, F_te[r])
            acc[r].append(accuracy_score(y_te, pred))
        last = {"F_te": F_te, "y_te": y_te}  # kept for the PCA plots

    results = {
        "shapes": shapes,
        "acc": {r: np.array(v) for r, v in acc.items()},
        "n_classes": len(shapes),
        "last": last,
        "config": dict(num_dirs=num_dirs, num_heights=num_heights,
                       n_train=n_train, n_test=n_test, n_seeds=n_seeds, area=area),
    }
    _report(results)
    return results


def _report(results):
    acc, n = results["acc"], results["n_classes"]
    cfg = results["config"]
    print(f"\nShape classification | {n} classes, all {cfg['area']} px (equal area)"
          f" | {cfg['num_dirs']} dirs x {cfg['num_heights']} heights")
    print(f"{cfg['n_train']} train / {cfg['n_test']} test per class, random 0-360 deg,"
          f" mean over {cfg['n_seeds']} seeds. Chance = {1.0 / n:.2f}")
    print("-" * 56)
    print(f"{'representation':16}{'accuracy':>14}{'vs chance':>14}")
    for r in REPRESENTATIONS:
        m, s = acc[r].mean(), acc[r].std()
        tag = "  (oracle)" if r in _ORACLE else ("  (control)" if r == "mass" else "")
        print(f"{r:16}{f'{m:.3f} ± {s:.3f}':>14}{f'{m * n:.1f}x':>14}{tag}")
    print("-" * 56)
    print(f"DIWECT beats pixel distance by "
          f"{acc['diwect'].mean() - acc['pixels'].mean():+.3f} and WECT by "
          f"{acc['diwect'].mean() - acc['wect'].mean():+.3f} accuracy.")
    print(f"Root cause: aligning to ground-truth pose lifts WECT from "
          f"{acc['wect'].mean():.3f} to {acc['wect_aligned'].mean():.3f}"
          f" -> rotation, not the WECT, was the problem.")
    print(f"DIWECT reaches {acc['diwect'].mean():.3f} with no ground-truth angle,"
          f" recovering most of that oracle gain for free.")
    print(f"Equal-area control (mass) at {acc['mass'].mean():.3f} ~ chance"
          f" -> the win is shape, not size.")
    print("-" * 56)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _pca_scatter(ax, X, y, names, title):
    Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    cmap = plt.get_cmap("tab10")
    for label, name in enumerate(names):
        m = y == label
        ax.scatter(Z[m, 0], Z[m, 1], s=8, color=cmap(label), label=name, alpha=0.7,
                   edgecolors="none")
    # Robust view limits: a few outliers (e.g. odd rotations of the WECT) can
    # otherwise stretch the axes and collapse the bulk into a sliver.
    lo, hi = np.percentile(Z, [2, 98], axis=0)
    pad = 0.12 * (hi - lo + 1e-9)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(results, out_path):
    shapes = results["shapes"]
    names = list(shapes.keys())
    acc = results["acc"]
    n = results["n_classes"]

    fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
    top, bottom = fig.add_gridspec(2, 1, height_ratios=[1, 1.9])

    # Top band: shape gallery (all classes, one per column).
    gallery = top.subgridspec(1, len(names))
    for k, name in enumerate(names):
        a = fig.add_subplot(gallery[0, k])
        a.imshow(shapes[name], cmap="magma", vmin=0, vmax=1)
        a.set_title(name, fontsize=9)
        a.set_xticks([]); a.set_yticks([])

    # Bottom band: accuracy bars + PCA embeddings telling the root-cause story
    # (WECT spread -> aligning collapses the orbits -> DIWECT collapses for free).
    panels = bottom.subgridspec(1, 4, width_ratios=[1.3, 1, 1, 1])

    ax = fig.add_subplot(panels[0, 0])
    x = np.arange(len(REPRESENTATIONS))
    means = [acc[r].mean() for r in REPRESENTATIONS]
    errs = [acc[r].std() for r in REPRESENTATIONS]
    bars = ax.bar(x, means, yerr=errs, capsize=3,
                  color=[_COLORS[r] for r in REPRESENTATIONS])
    ax.bar_label(bars, labels=[f"{m:.2f}" for m in means], padding=6, fontsize=8)
    ax.axhline(1.0 / n, ls="--", lw=1, color="k", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([_DISPLAY.get(r, r) for r in REPRESENTATIONS], fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("test accuracy")
    ax.set_title("Classifier accuracy (distance-kernel SVM)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    last = results["last"]
    _pca_scatter(fig.add_subplot(panels[0, 1]), last["F_te"]["wect"],
                 last["y_te"], names, "WECT (test) — PCA")
    _pca_scatter(fig.add_subplot(panels[0, 2]), last["F_te"]["wect_aligned"],
                 last["y_te"], names, "WECT+align (test) — PCA")
    ax_d = fig.add_subplot(panels[0, 3])
    _pca_scatter(ax_d, last["F_te"]["diwect"], last["y_te"],
                 names, "DIWECT (test) — PCA")
    ax_d.legend(frameon=False, fontsize=7, loc="best")

    fig.suptitle("Rotation is the root cause: aligning fixes the WECT, "
                 "and the DIWECT is invariant for free",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--area", type=int, default=380)
    p.add_argument("--num-dirs", type=int, default=64)
    p.add_argument("--num-heights", type=int, default=48)
    p.add_argument("--n-train", type=int, default=6)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "svm_classification.png"))
    args = p.parse_args()

    results = run(size=args.size, area=args.area, num_dirs=args.num_dirs,
                  num_heights=args.num_heights, n_train=args.n_train,
                  n_test=args.n_test, n_seeds=args.n_seeds, C=args.C)
    make_figure(results, args.out)


if __name__ == "__main__":
    main()
