"""ModelNet10 loader for the real-world 3D DIWECT-vs-WECT experiment.

ModelNet10 (Princeton) is the canonical CAD-mesh classification benchmark: 10
household-object classes as OFF triangle meshes, already split into train/test
and posed in a consistent upright orientation. That canonical pose is exactly
the interesting regime for this comparison -- when orientation is *signal*, the
directional WECT should have an edge; when we rotate objects about the vertical
(a realistic nuisance), the direction-integrated DIWECT should hold up better.

Because raw ModelNet meshes vary from hundreds to hundreds of thousands of
faces, and a unit-weight WECT's magnitude scales with the simplex count, we
resample every object to a uniform voxel resolution (which doubles as the raw
``voxels`` baseline) and rebuild a watertight surface mesh with marching cubes
-- the same authoring path as the synthetic experiment.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from core.meshes import Mesh, occ_to_mesh, voxelize

MODELNET10_URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
MODELNET10_DIR = "ModelNet10"

MODELNET10_CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


def _list_split(root: str, split: str) -> Tuple[List[str], np.ndarray]:
    files, labels = [], []
    for label, cls in enumerate(MODELNET10_CLASSES):
        pattern = os.path.join(root, cls, split, f"{cls}_*.off")
        for path in sorted(glob.glob(pattern)):
            files.append(path)
            labels.append(label)
    return files, np.asarray(labels, dtype=np.int64)


def load_modelnet10(cache_dir: str, val_frac: float = 0.15, seed: int = 0) -> dict:
    """Return file paths, labels and a (train, val, test) index split.

    ModelNet10 ships train/ and test/ folders; we carve a stratified validation
    set out of the training split so hyper-parameter tuning never touches test.
    """
    root = os.path.join(cache_dir, MODELNET10_DIR)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Download (~473 MB) and unzip with:\n"
            f"  curl -L -o {cache_dir}/ModelNet10.zip {MODELNET10_URL}\n"
            f"  unzip {cache_dir}/ModelNet10.zip -d {cache_dir}"
        )
    tr_files, tr_labels = _list_split(root, "train")
    te_files, te_labels = _list_split(root, "test")

    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(tr_labels):
        idx = np.where(tr_labels == c)[0]
        rng.shuffle(idx)
        n_va = int(round(val_frac * len(idx)))
        va_idx.append(idx[:n_va])
        tr_idx.append(idx[n_va:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)

    files = tr_files + te_files
    labels = np.concatenate([tr_labels, te_labels])
    te_idx = np.arange(len(tr_files), len(files))
    return dict(files=files, labels=labels, split=(tr_idx, va_idx, te_idx),
                n_classes=len(MODELNET10_CLASSES), classes=MODELNET10_CLASSES,
                name="modelnet10")


def load_normalized_mesh(path: str) -> Mesh:
    """Load an OFF mesh, center it and scale so ``max |coord| == 1`` (aspect kept)."""
    m = trimesh.load(path, process=False, force="mesh")
    verts = np.asarray(m.vertices, np.float32)
    verts = verts - verts.mean(0)
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale
    return Mesh(verts.astype(np.float32), np.asarray(m.faces, np.int64))


def resample_mesh(mesh: Mesh, grid: int, half: float = 1.45) -> Mesh:
    """Voxelize to a uniform ``grid`` occupancy and rebuild a marching-cubes mesh.

    Returns a watertight, uniformly tessellated mesh whose orientation can then
    be rotated *exactly* -- unlike re-voxelizing a rotated raw object, which
    aliases and would destroy the DIWECT's rotation invariance. The world box
    (``half``) is wide enough that an azimuthal rotation of the unit shape
    (extent up to sqrt(2)) still fits, so canonical and rotated samples share
    one voxel pitch.
    """
    occ = voxelize(mesh.verts, mesh.faces, grid=grid, half=half)
    if occ.sum() == 0:  # degenerate (e.g. a flat sheet) -> keep a single voxel
        occ[grid // 2, grid // 2, grid // 2] = True
    return occ_to_mesh(occ)
