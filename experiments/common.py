"""Shared code for the synthetic experiments (Figures 1-3): paths, plot style,
the Equal Area Shapes 2D (EAS2D) dataset, and a thin wrapper around pyECT's WECT.

Everything a notebook in this directory reads or writes lives under ``ROOT``
(this directory): ``data/`` for inputs and caches, ``figures/`` for outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from pyect import WECT, sample_directions_2d, weighted_cubical
from scipy.ndimage import convolve

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def figure_dir(name: str) -> Path:
    """``figures/<name>/``, created if needed (names follow the paper's figure folders)."""
    path = FIGURES / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Plot style: the WECT is always red and the DIWECT always blue
# --------------------------------------------------------------------------- #

COLORS: Dict[str, str] = {
    "wect": "#d1495b",
    "diwect": "#30638e",
    "baseline": "#d1495b",     # not rotation invariant
    "invariant": "#30638e",    # exactly rotation invariant
    "oracle": "#2a9d8f",       # needs the ground-truth pose
    "control": "#c9ced6",      # chance-level control
    "dim control": "#e09f3e",  # dimension control
}
DISPLAY = {"wect": "WECT", "diwect": "DIWECT"}


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.alpha": 0.25, "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def color(name: str) -> str:
    return COLORS.get(name, "#555555")


def label(name: str) -> str:
    return DISPLAY.get(name, name)


def bar_labels(ax, bars, fmt="{:.2f}", fontsize=7.5, padding=2):
    ax.bar_label(bars, labels=[fmt.format(b.get_height()) for b in bars],
                 padding=padding, fontsize=fontsize)


def box_plot(ax, samples, positions, colors, width=0.6, alpha=1.0, vert=True, points=True,
             ink="#222222", seed=0):
    bp = ax.boxplot(samples, positions=positions, widths=width, orientation="vertical" if vert else "horizontal",
                    patch_artist=True, showfliers=not points, manage_ticks=False,
                    medianprops=dict(color=ink, lw=1.4), whiskerprops=dict(color=ink, lw=0.8),
                    capprops=dict(color=ink, lw=0.8),
                    flierprops=dict(marker="o", ms=2.5, mfc="none", mec=ink, mew=0.6),
                    boxprops=dict(edgecolor=ink, lw=0.8))
    span = np.ptp(np.concatenate([np.asarray(v, float) for v in samples])) or 1.0
    for patch, med, c, vals in zip(bp["boxes"], bp["medians"], colors, samples):
        rgba = mpl.colors.to_rgba(c, alpha)
        patch.set_facecolor(rgba)
        # the median must stay visible on its fill: white on dark fills, ink on light ones;
        # a box too thin to show fill around a white line keeps the ink median
        r, g, b = (alpha * np.array(rgba[:3]) + (1 - alpha))   # composited on white
        q1, q3 = np.percentile(vals, [25, 75])
        if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.38 and (q3 - q1) > 0.03 * span:
            med.set_color("white")
    if points:
        rng = np.random.default_rng(seed)
        for pos, vals in zip(positions, samples):
            vals = np.asarray(vals)
            jit = pos + rng.uniform(-0.18, 0.18, len(vals)) * width
            xy = (jit, vals) if vert else (vals, jit)
            ax.scatter(*xy, s=6, color=ink, alpha=0.7, lw=0, zorder=3)
    return bp


def median_labels(ax, samples, positions, fmt="{:.2f}", fontsize=7.5, pad=0.015):
    """Write each box's median just above its highest sample (vertical boxes)."""
    for pos, vals in zip(positions, samples):
        vals = np.asarray(vals)
        top = vals.max()
        ax.annotate(fmt.format(np.median(vals)), (pos, top + pad), ha="center", va="bottom", fontsize=fontsize)


def chance_line(ax, n_classes, text="chance"):
    ax.axhline(1.0 / n_classes, ls="--", lw=1, color="k", zorder=0)
    ax.annotate(text, xy=(0.995, 1.0 / n_classes), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.5, color="k")


def save_pdf(fig, folder: str, stem: str) -> None:
    """Write ``figures/<folder>/<stem>.pdf`` and close the figure."""
    import matplotlib.pyplot as plt

    path = figure_dir(folder) / f"{stem}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# EAS2D: five asymmetric polygons rasterized to exactly the same pixel area
# --------------------------------------------------------------------------- #

# Polygons in a centred frame (x right, y up), before area matching.
SHAPE_POLYGONS: Dict[str, List[Tuple[float, float]]] = {
    "bar": [(-26, -4), (26, -4), (26, 4), (-26, 4)],                         # rod
    "ell": [(-5, -18), (3, -18), (3, 10), (16, 10), (16, 18), (-5, 18)],     # L
    "tee": [(-16, -18), (16, -18), (16, -10), (4, -10),
            (4, 18), (-4, 18), (-4, -10), (-16, -10)],                       # T
    "plus": [(-4, -18), (4, -18), (4, -4), (18, -4), (18, 4), (4, 4),
             (4, 18), (-4, 18), (-4, 4), (-18, 4), (-18, -4), (-4, -4)],     # +
    "tri": [(-20, -14), (20, -14), (0, 20)],                                 # triangle
}

_NEIGHBORS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)


def _rasterize(vertices: Sequence[Tuple[float, float]], size: int, scale: float) -> np.ndarray:
    center = (size - 1) / 2.0
    img = Image.new("1", (size, size), 0)
    pts = [(center + scale * x, center - scale * y) for x, y in vertices]  # y up
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def _erode_to(mask: np.ndarray, target: int) -> np.ndarray:
    """Remove the most exposed on-pixel (fewest on-neighbours) until `target` remain."""
    mask = mask.copy()
    while int(mask.sum()) > target:
        nb = convolve(mask.astype(np.int8), _NEIGHBORS, mode="constant")
        nb = np.where(mask, nb, 99)  # ignore off-pixels
        mask.flat[int(np.argmin(nb))] = False
    return mask


def make_shape(vertices, size: int = 64, area: int = 380) -> np.ndarray:
    """Grow the polygon until it covers >= `area` pixels, then erode to exactly `area`."""
    scale = 0.3
    while int(_rasterize(vertices, size, scale).sum()) < area:
        scale += 0.02
    return _erode_to(_rasterize(vertices, size, scale), area).astype(np.float32)


def make_shape_library(size: int = 64, area: int = 380) -> Dict[str, np.ndarray]:
    return {name: make_shape(poly, size=size, area=area) for name, poly in SHAPE_POLYGONS.items()}


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Nearest-neighbour rotation about the image centre (no resampling blur)."""
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0)
    r = TF.rotate(t, float(angle_deg), interpolation=TF.InterpolationMode.NEAREST, expand=False)
    return r.squeeze(0).clamp_min(0.0).numpy()


# --------------------------------------------------------------------------- #
# WECT of an image (pyECT), and the DIWECT as its mean over directions
# --------------------------------------------------------------------------- #


class WECTTransform:
    """pyECT's WECT of a 2D image's weighted cubical complex.

    Directions are pyECT's evenly spaced ``theta_d = 2 pi d / D`` (in order), so a
    rotation by ``2 pi k / D`` cyclically shifts the direction axis by k rows. The
    column mean of this matrix (``diwect``) is the direction-averaged WECT.
    """

    def __init__(self, num_dirs: int = 64, num_heights: int = 64) -> None:
        self.dirs = sample_directions_2d(num_dirs)
        self.num_dirs = num_dirs
        self.num_heights = int(num_heights)
        self._module = WECT(self.dirs, self.num_heights)

    def _matrix(self, image) -> torch.Tensor:
        img = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        with torch.no_grad():
            return self._module(weighted_cubical(img))

    def wect(self, image) -> np.ndarray:
        return self._matrix(image).cpu().numpy()                 # (D, H)

    def diwect(self, image) -> np.ndarray:
        return self._matrix(image).mean(dim=0).cpu().numpy()     # (H,)


def median_gamma(sqdists: np.ndarray) -> float:
    """RBF bandwidth from the median heuristic on a squared-distance matrix."""
    positive = sqdists[sqdists > 0]
    if positive.size == 0:
        return 1.0
    return 1.0 / (float(np.median(positive)) + 1e-12)
