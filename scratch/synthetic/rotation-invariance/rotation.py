from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.wect import WECTTransform


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def make_l_shape(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    # vertical bar (bright)
    img[c - 18 : c + 18, c - 6 : c + 2] = 1.0
    # horizontal foot (dimmer) -> breaks all symmetry
    img[c + 10 : c + 18, c - 6 : c + 14] = 0.6
    return img


def make_disk(size: int = 64, radius: int = 16) -> np.ndarray:
    c = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - c) ** 2 + (y - c) ** 2 <= radius * radius
    return mask.astype(np.float32)


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
    r = TF.rotate(
        t,
        float(angle_deg),
        interpolation=TF.InterpolationMode.NEAREST,
        expand=False,
    )
    return r.squeeze(0).clamp_min(0.0).numpy()


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def relative_l2(x: np.ndarray, ref: np.ndarray) -> float:
    denom = np.linalg.norm(ref.ravel())
    return float(np.linalg.norm((x - ref).ravel()) / (denom + 1e-12))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(
    size: int = 64,
    num_dirs: int = 128,
    num_heights: int = 64,
    angle_step: float = 3.0,
    complex_type: str = "cubical",
):
    transform = WECTTransform(
        num_dirs=num_dirs,
        num_heights=num_heights,
        complex_type=complex_type,
    )

    angles = np.arange(0.0, 360.0, angle_step)
    base = make_l_shape(size)

    # Reference descriptors: the un-rotated L (angle 0, no interpolation).
    ref_wect = transform.wect(base)
    ref_diwect = transform.diwect(base)

    wect_drift = np.empty(len(angles))
    diwect_drift = np.empty(len(angles))
    for i, ang in enumerate(angles):
        rot = rotate_image(base, ang)
        wect_drift[i] = relative_l2(transform.wect(rot), ref_wect)
        diwect_drift[i] = relative_l2(transform.diwect(rot), ref_diwect)

    # compare against disk
    disk = make_disk(size)
    cross_diwect = relative_l2(transform.diwect(disk), ref_diwect)
    cross_wect = relative_l2(transform.wect(disk), ref_wect)

    results = {
        "angles": angles,
        "wect_drift": wect_drift,
        "diwect_drift": diwect_drift,
        "cross_diwect": cross_diwect,
        "cross_wect": cross_wect,
        "base": base,
    }
    _report(results, num_dirs, num_heights, complex_type)
    return results


def _report(results, num_dirs, num_heights, complex_type):
    wd, dd = results["wect_drift"], results["diwect_drift"]
    print(f"\nRotation experiment (L-shape) | {num_dirs} dirs x {num_heights} heights"
          f" | {complex_type} complex")
    print("-" * 64)
    print(f"{'':22}{'WECT':>12}{'DIWECT':>12}{'ratio W/D':>14}")
    print(f"{'mean drift vs 0deg':22}{wd.mean():>12.4f}{dd.mean():>12.4f}"
          f"{wd.mean() / (dd.mean() + 1e-12):>14.1f}")
    print(f"{'max  drift vs 0deg':22}{wd.max():>12.4f}{dd.max():>12.4f}"
          f"{wd.max() / (dd.max() + 1e-12):>14.1f}")
    print("-" * 64)
    print("Discriminability yardstick (drift to a *different* shape, a disk):")
    print(f"  WECT   L->disk distance: {results['cross_wect']:.4f}"
          f"   (rotational max {wd.max():.4f})")
    print(f"  DIWECT L->disk distance: {results['cross_diwect']:.4f}"
          f"   (rotational max {dd.max():.4f})")
    ratio = results["cross_diwect"] / (dd.max() + 1e-12)
    print(f"  -> A different shape moves the DIWECT {ratio:.1f}x more than the"
          " worst rotation of the L does.")
    print("-" * 64)


def make_figure(results, out_path: str):
    angles = results["angles"]
    fig = plt.figure(figsize=(11, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    # Left: drift curves.
    ax = fig.add_subplot(gs[:, :2])
    ax.plot(angles, results["wect_drift"], color="#d1495b", lw=2, label="WECT")
    ax.plot(angles, results["diwect_drift"], color="#30638e", lw=2, label="DIWECT")
    ax.axhline(results["cross_diwect"], color="#30638e", ls=":", lw=1.2,
               label="DIWECT L→disk (different shape)")
    ax.set_xlabel("rotation angle (degrees)")
    ax.set_ylabel("relative L2 drift from 0°")
    ax.set_title("Descriptor drift under rotation")
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: the L at four angles, to show it really is being rotated.
    for k, ang in enumerate([0, 45, 90, 135]):
        a = fig.add_subplot(gs[k // 2, 2 + k % 2])
        a.imshow(rotate_image(results["base"], ang), cmap="magma", vmin=0, vmax=1)
        a.set_title(f"{ang}°", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle("DIWECT is rotation-stable where the WECT is not",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--num-dirs", type=int, default=128)
    p.add_argument("--num-heights", type=int, default=64)
    p.add_argument("--angle-step", type=float, default=3.0)
    p.add_argument("--complex-type", choices=["cubical", "freudenthal"], default="cubical")
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_invariance.png"),
    )
    args = p.parse_args()

    results = run(
        size=args.size,
        num_dirs=args.num_dirs,
        num_heights=args.num_heights,
        angle_step=args.angle_step,
        complex_type=args.complex_type,
    )
    make_figure(results, args.out)


if __name__ == "__main__":
    main()
