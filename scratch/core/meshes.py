"""Synthetic 3D shape meshes for the DIWECT-vs-WECT comparison.

The 3D analogue of ``core.shapes``.  We build a small library of genus-0 solids
that share the *same* volume but differ in how their mass is distributed along
the coordinate axes (a rod, a plate, a bent-L, a tee, a 3-axis cross).  Because
the classes are volume-matched, a classifier cannot cheat on size -- it has to
read the *shape*, which is exactly what the WECT / DIWECT summarize.

Each class is authored on a voxel grid, resized to an exact target voxel count
(equal-volume control, the 3D version of the equal-area control in the 2D
experiments), and converted to a watertight triangle mesh with marching cubes.
Samples are produced by applying an *exact* rotation matrix to the mesh
vertices -- there is no grid interpolation anywhere, so the rotation test is
free of the max-extension interpolation confound that bites the 2D pipeline.
Two representations are then derived from the very same rotated mesh:

    * ``voxels``  -- a fresh occupancy voxelization (the raw-geometry baseline),
    * a weighted simplicial ``Complex`` -- fed to the WECT / DIWECT.
"""
from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Tuple

import numpy as np
import torch
import trimesh
from scipy.ndimage import distance_transform_edt
from skimage import measure

from pyect.tensor_complex import Complex


class Mesh(NamedTuple):
    verts: np.ndarray  # (V, 3) float32, centered, max |coord| == 1
    faces: np.ndarray  # (F, 3) int64


# --------------------------------------------------------------------------- #
# Shape authoring (voxel occupancy in a centered grid)
# --------------------------------------------------------------------------- #
def _box(grid: int, half: Tuple[float, float, float]) -> np.ndarray:
    """Axis-aligned solid box with half-extents given as fractions of the grid."""
    c = (grid - 1) / 2.0
    ax = [np.abs(np.arange(grid) - c) for _ in range(3)]
    hx, hy, hz = (h * c for h in half)
    return (
        (ax[0][:, None, None] <= hx)
        & (ax[1][None, :, None] <= hy)
        & (ax[2][None, None, :] <= hz)
    )


def _rod(grid: int) -> np.ndarray:  # one long axis
    return _box(grid, (0.28, 0.28, 0.95))


def _plate(grid: int) -> np.ndarray:  # two long axes, thin
    return _box(grid, (0.95, 0.95, 0.16))


def _ell(grid: int) -> np.ndarray:  # bent rod (L) in the x-z plane
    return _box(grid, (0.30, 0.30, 0.9)) | _box_off(
        grid, (0.9, 0.30, 0.30), (0.55, 0.0, -0.55))


def _tee(grid: int) -> np.ndarray:  # T in the x-z plane
    return _box(grid, (0.95, 0.28, 0.28)) | _box_off(
        grid, (0.28, 0.28, 0.9), (0.0, 0.0, -0.55))


def _cross(grid: int) -> np.ndarray:  # arms along all three axes
    return (
        _box(grid, (0.95, 0.22, 0.22))
        | _box(grid, (0.22, 0.95, 0.22))
        | _box(grid, (0.22, 0.22, 0.95))
    )


def _box_off(grid: int, half, offset) -> np.ndarray:
    """A box whose center is shifted by ``offset`` (fractions of the half-grid)."""
    c = (grid - 1) / 2.0
    coords = [np.arange(grid) - c - o * c for o in offset]
    hx, hy, hz = (h * c for h in half)
    return (
        (np.abs(coords[0])[:, None, None] <= hx)
        & (np.abs(coords[1])[None, :, None] <= hy)
        & (np.abs(coords[2])[None, None, :] <= hz)
    )


SHAPE_VOXELS: Dict[str, Callable[[int], np.ndarray]] = {
    "rod": _rod,
    "plate": _plate,
    "ell": _ell,
    "tee": _tee,
    "cross": _cross,
}


def resize_to_volume(occ: np.ndarray, target: int) -> np.ndarray:
    """Erode/dilate a binary volume from its boundary to an exact voxel count."""
    occ = occ.copy()
    cur = int(occ.sum())
    if cur > target:  # peel outermost voxels first (smallest distance to outside)
        d = distance_transform_edt(occ)
        on = np.argwhere(occ)
        order = on[np.argsort(d[occ], kind="stable")]
        occ[tuple(order[: cur - target].T)] = False
    elif cur < target:  # accrete nearest background voxels first
        d = distance_transform_edt(~occ)
        off = np.argwhere(~occ)
        order = off[np.argsort(d[~occ], kind="stable")]
        occ[tuple(order[: target - cur].T)] = True
    return occ


# --------------------------------------------------------------------------- #
# Mesh construction
# --------------------------------------------------------------------------- #
def occ_to_mesh(occ: np.ndarray) -> Mesh:
    """Watertight surface mesh of an occupancy volume via marching cubes."""
    padded = np.pad(occ.astype(np.float32), 1)  # pad so the surface closes
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = verts - verts.mean(0)
    verts = verts / np.abs(verts).max()  # canonical: max |coord| == 1
    return Mesh(verts.astype(np.float32), faces.astype(np.int64))


def make_mesh_library(grid: int = 32, volume: int = 3200) -> Dict[str, Mesh]:
    """One volume-matched canonical mesh per shape class."""
    return {
        name: occ_to_mesh(resize_to_volume(build(grid), volume))
        for name, build in SHAPE_VOXELS.items()
    }


# --------------------------------------------------------------------------- #
# Sampling: exact rotation + representations
# --------------------------------------------------------------------------- #
def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3x3 rotation matrix (QR of a Gaussian, det fixed +1)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))  # make the decomposition unique
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def rotate(verts: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return verts @ rot.T


def mesh_to_complex(verts: np.ndarray, faces: np.ndarray,
                    device: torch.device | None = None) -> Complex:
    """In-memory weighted simplicial complex (unit weights) from a triangle mesh."""
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vc = np.asarray(m.vertices, np.float32)
    vc = vc - vc.mean(0)  # WECT scales heights by distance-from-origin -> center
    edges = np.asarray(m.edges_unique, np.int64)
    tris = np.asarray(m.faces, np.int64)
    v = (torch.tensor(vc), torch.ones(len(vc)))
    e = (torch.tensor(edges), torch.ones(len(edges)))
    t = (torch.tensor(tris), torch.ones(len(tris)))
    return Complex(v, e, t, device=device)


def voxelize(verts: np.ndarray, faces: np.ndarray,
             grid: int = 24, half: float = 1.5) -> np.ndarray:
    """Filled occupancy of a mesh in a fixed [-half, half]^3 world box.

    The box is wide enough that any rotation of a unit-normalized shape still
    fits (the widest class, the plate, reaches ~1.42 on an axis), so
    orientation genuinely scrambles the occupancy pattern -- the nuisance the
    raw-voxel baseline has to overcome.
    """
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.vertices = m.vertices - m.vertices.mean(0)
    pitch = 2.0 * half / grid
    vg = m.voxelized(pitch=pitch)
    try:
        vg = vg.fill()
    except Exception:  # non-watertight after rotation; surface voxels are fine
        pass
    idx = np.floor((vg.points + half) / pitch).astype(int)
    ok = ((idx >= 0) & (idx < grid)).all(1)
    idx = idx[ok]
    out = np.zeros((grid, grid, grid), bool)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out
