from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "numdirs_sweep.csv")
OUT = os.path.join(HERE, "results", "summary.png")

ROWS = [
    ("svm", "shapes", "SVM\nequal-area shapes"),
    ("mlp", "shapes", "MLP\nequal-area shapes"),
    ("real", "bloodmnist", "BloodMNIST\nResNet18 + topo"),
    ("real", "galaxy10", "Galaxy10\nResNet18 + topo"),
]
CANON = ["pixels", "wect", "wect_aligned", "diwect", "pixels+wect", "pixels+diwect"]
COLORS = {"wect": "#d1495b", "wect_aligned": "#2a9d8f", "diwect": "#30638e",
          "pixels+wect": "#9b5de5", "pixels+diwect": "#f4a261", "pixels": "#8d99ae"}
DISPLAY = {"wect_aligned": "wect+align", "pixels+wect": "pix+WECT",
           "pixels+diwect": "pix+DIWECT"}
CHANCE = {"shapes": 1 / 5, "bloodmnist": 1 / 8, "galaxy10": 1 / 10}
PIX = "pixels"


def agg(df):
    return (df.groupby(["experiment", "dataset", "representation", "num_dirs"])
              .agg(acc=("accuracy", "mean"), sd=("accuracy", "std"),
                   params=("head_params", "mean"), fdim=("feature_dim", "mean"))
              .reset_index())


def size_col(exp):
    return "fdim" if exp == "svm" else "params"


def best_row(sub):
    return sub.loc[sub["acc"].idxmax()]


def ceiling_size(sub, scol, tol=0.01):
    best = sub["acc"].max()
    ok = sub[sub["acc"] >= best - tol].sort_values("num_dirs")
    return float(ok.iloc[0][scol])


def reps_of(block):
    return [r for r in CANON if r in set(block["representation"])]


def pixels_stats(block, scol):
    p = block[block["representation"] == PIX]
    if p.empty:
        return None
    row = p.iloc[0]
    return float(row["acc"]), float(row["sd"]), float(row[scol])


# --------------------------------------------------------------------------- #
def panel_headtohead(ax, block, exp, dset, reps):
    best = {r: best_row(block[block["representation"] == r]) for r in reps}
    x = np.arange(len(reps))
    ax.bar(x, [best[r]["acc"] for r in reps], yerr=[best[r]["sd"] for r in reps],
           capsize=3, color=[COLORS[r] for r in reps])
    for xi, r in zip(x, reps):
        lab = "" if r == PIX else f"D{int(best[r]['num_dirs'])}"
        ax.text(xi, best[r]["acc"] + 0.02, lab, ha="center", va="bottom", fontsize=7)
    ax.axhline(CHANCE[dset], ls="--", lw=1, color="k", label="chance")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY.get(r, r) for r in reps],
                                         fontsize=7, rotation=20)
    ax.set_ylim(0, 1.12); ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_vs_dirs(ax, block, exp, dset, reps):
    for r in reps:
        s = block[block["representation"] == r].sort_values("num_dirs")
        ax.errorbar(s["num_dirs"], s["acc"], yerr=s["sd"], color=COLORS[r], lw=2,
                    marker="o", ms=4, capsize=2, label=DISPLAY.get(r, r))
        b = best_row(s)
        ax.scatter([b["num_dirs"]], [b["acc"]], s=90, facecolors="none",
                   edgecolors=COLORS[r], lw=1.8, zorder=5)
    pix = pixels_stats(block, size_col(exp))
    if pix is not None:
        acc, sd, _ = pix
        ax.axhline(acc, ls=":", lw=1.6, color=COLORS[PIX], label="pixels", zorder=1)
        ax.axhspan(acc - sd, acc + sd, color=COLORS[PIX], alpha=0.10, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(block[block["num_dirs"] > 0]["num_dirs"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of directions"); ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_vs_size(ax, block, exp, reps):
    scol = size_col(exp)
    for r in reps:
        s = block[block["representation"] == r].sort_values("num_dirs")
        ax.plot(s[scol], s["acc"], color=COLORS[r], lw=2, marker="o", ms=4,
                label=DISPLAY.get(r, r))
        b = best_row(s)
        ax.scatter([b[scol]], [b["acc"]], s=90, facecolors="none",
                   edgecolors=COLORS[r], lw=1.8, zorder=5)
    pix = pixels_stats(block, scol)
    if pix is not None:
        acc, _, size = pix
        ax.scatter([size], [acc], s=80, marker="D", color=COLORS[PIX],
                   edgecolors="k", lw=0.6, zorder=6, label="pixels")
    ax.set_xscale("log")
    ax.set_xlabel("feature dim (SVM)" if exp == "svm" else "trainable head params")
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_ceiling_size(ax, block, exp, reps):
    scol = size_col(exp)
    sizes = {r: ceiling_size(block[block["representation"] == r], scol) for r in reps}
    x = np.arange(len(reps))
    ax.bar(x, [sizes[r] for r in reps], color=[COLORS[r] for r in reps])
    for xi, r in zip(x, reps):
        ax.text(xi, sizes[r] * 1.15, f"{int(sizes[r]):,}", ha="center", va="bottom",
                fontsize=7, rotation=0)
    if "wect" in sizes and "diwect" in sizes and sizes["diwect"] > 0:
        ratio = sizes["wect"] / sizes["diwect"]
        ax.text(0.5, 0.92, f"WECT / DIWECT = {ratio:.1f}x", transform=ax.transAxes,
                ha="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0c97a"))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY.get(r, r) for r in reps],
                                         fontsize=7, rotation=20)
    ax.set_ylabel("feature dim" if exp == "svm" else "head params")
    ax.set_ylim(top=max(sizes.values()) * 3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    g = agg(pd.read_csv(CSV))
    nrows = len(ROWS)
    fig, axes = plt.subplots(nrows, 4, figsize=(19, 4.3 * nrows),
                             constrained_layout=True)
    col_titles = ["Best-HP head-to-head\n(label = best #directions)",
                  "Accuracy vs #directions\n(○ = best; WECT plateaus, DIWECT climbs)",
                  "Accuracy vs head size\n(efficiency: DIWECT far left)",
                  "Head size to reach ceiling\n(within 1% of own best)"]

    for ri, (exp, dset, title) in enumerate(ROWS):
        block = g[(g["experiment"] == exp) & (g["dataset"] == dset)]
        bar_reps = reps_of(block)                       # includes pixels (as a bar)
        sweep_reps = [r for r in bar_reps if r != PIX]  # excludes pixels (line panels)
        panel_headtohead(axes[ri, 0], block, exp, dset, bar_reps)
        panel_vs_dirs(axes[ri, 1], block, exp, dset, sweep_reps)
        panel_vs_size(axes[ri, 2], block, exp, sweep_reps)
        panel_ceiling_size(axes[ri, 3], block, exp, bar_reps)
        # Row label to the left of the first panel.
        axes[ri, 0].text(-0.34, 0.5, title, transform=axes[ri, 0].transAxes,
                         ha="center", va="center", rotation=90, fontsize=11,
                         fontweight="bold")
        if ri == 0:
            for ci in range(4):
                axes[ri, ci].set_title(col_titles[ci], fontsize=10)

    fig.suptitle("DIWECT vs WECT at each representation's best hyperparameters "
                 "(tuned lr/wd + best #directions): parity/boost on accuracy, "
                 "far smaller head",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {OUT}")


if __name__ == "__main__":
    main()
