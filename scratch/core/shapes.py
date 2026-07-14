from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from scipy.ndimage import convolve

Vertices = Sequence[Tuple[float, float]]

# Polygons in a centered coordinate frame (x right, y up), before area matching.
# Each is asymmetric enough that rotation scrambles a pixel/WECT representation
SHAPE_POLYGONS: Dict[str, List[Tuple[float, float]]] = {
    "bar": [(-26, -4), (26, -4), (26, 4), (-26, 4)],  # elongated rod
    "ell": [(-5, -18), (3, -18), (3, 10), (16, 10), (16, 18), (-5, 18)],  # L
    "tee": [(-16, -18), (16, -18), (16, -10), (4, -10),
            (4, 18), (-4, 18), (-4, -10), (-16, -10)],  # T
    "plus": [(-4, -18), (4, -18), (4, -4), (18, -4), (18, 4), (4, 4),
             (4, 18), (-4, 18), (-4, 4), (-18, 4), (-18, -4), (-4, -4)],  # +
    "tri": [(-20, -14), (20, -14), (0, 20)],  # asymmetric triangle
}

_NEIGHBORS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)


def _rasterize(vertices: Vertices, size: int, scale: float) -> np.ndarray:
    center = (size - 1) / 2.0
    img = Image.new("1", (size, size), 0)
    pts = [(center + scale * x, center - scale * y) for x, y in vertices]  # y up
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def _erode_to(mask: np.ndarray, target: int) -> np.ndarray:
    mask = mask.copy()
    while int(mask.sum()) > target:
        nb = convolve(mask.astype(np.int8), _NEIGHBORS, mode="constant")
        nb = np.where(mask, nb, 99)  # ignore off-pixels
        mask.flat[int(np.argmin(nb))] = False
    return mask


def make_shape(vertices: Vertices, size: int = 64, area: int = 380) -> np.ndarray:
    scale = 0.3
    while int(_rasterize(vertices, size, scale).sum()) < area:
        scale += 0.02
    return _erode_to(_rasterize(vertices, size, scale), area).astype(np.float32)


def make_shape_library(size: int = 64, area: int = 380) -> Dict[str, np.ndarray]:
    return {name: make_shape(poly, size=size, area=area)
            for name, poly in SHAPE_POLYGONS.items()}


def apply_intensity(img: np.ndarray, rng: np.random.Generator,
                    scale_range: Tuple[float, float] = (0.4, 1.0),
                    jitter: float = 0.1) -> np.ndarray:
    out = np.asarray(img, dtype=np.float32).copy()
    on = out > 0
    c = rng.uniform(*scale_range)
    out[on] = out[on] * c * (1.0 + rng.uniform(-jitter, jitter, size=int(on.sum())))
    return out


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0)
    r = TF.rotate(
        t, float(angle_deg),
        interpolation=TF.InterpolationMode.NEAREST, expand=False,
    )
    return r.squeeze(0).clamp_min(0.0).numpy()
