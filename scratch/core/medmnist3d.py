"""MedMNIST v2 3D loaders for the DIWECT-vs-WECT comparison.

These 28^3 medical volumes are the natural "biological" counterpoint to
ModelNet: several are *shape* datasets (binary segmentation masks) whose objects
have no canonical orientation, so -- unlike a chair -- orientation is not class
signal and the direction-integrated DIWECT should be competitive with the WECT.

    adrenalmnist3d  adrenal-gland shape masks    normal vs hyperplasia   (binary)
    vesselmnist3d   intracranial vessel masks    vessel vs aneurysm      (binary)
    synapsemnist3d  synapse volumes              inhibitory vs excitatory
    nodulemnist3d   lung-nodule CT               benign vs malignant     (grayscale)
    organmnist3d    organ CT (11 classes)        canonical anatomical pose (grayscale)
    fracturemnist3d rib-fracture CT (3 classes)

Every sample becomes an occupancy grid (a threshold for the grayscale sets), so
it feeds the same marching-cubes -> mesh -> WECT path as the other 3D
experiments.
"""
from __future__ import annotations

import numpy as np

MEDMNIST3D = [
    "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",
    "nodulemnist3d", "organmnist3d", "fracturemnist3d",
]
# Datasets that already ship as binary segmentation masks (no thresholding).
# NB: synapsemnist3d looks mask-like but is a *grayscale* electron-microscopy
# volume (256 levels, ~99.6% of voxels nonzero) -- it must be thresholded, not
# treated as a mask, or every sample collapses to a near-solid cube.
_SHAPE_MASK = {"adrenalmnist3d", "vesselmnist3d"}


def load_medmnist3d(name: str, cache_dir: str) -> dict:
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    vols, labels, sizes = [], [], {}
    for split in ["train", "val", "test"]:
        d = DataClass(split=split, download=True, root=cache_dir)
        vols.append(np.asarray(d.imgs))
        labels.append(np.asarray(d.labels).flatten())
        sizes[split] = len(d.imgs)
    volumes = np.concatenate(vols).astype(np.uint8)  # (N, 28, 28, 28)
    labels = np.concatenate(labels).astype(np.int64)
    n_tr, n_va = sizes["train"], sizes["val"]
    split = (np.arange(n_tr), np.arange(n_tr, n_tr + n_va),
             np.arange(n_tr + n_va, len(labels)))
    return dict(volumes=volumes, labels=labels, split=split,
                n_classes=len(info["label"]), classes=list(info["label"].values()),
                is_mask=name in _SHAPE_MASK, name=name)


def volume_to_occupancy(vol: np.ndarray, is_mask: bool, percentile: float = 55.0) -> np.ndarray:
    """Binary occupancy of a volume. Masks threshold at >0; grayscale volumes at
    an intensity percentile (keeps the denser interior of an organ/nodule)."""
    vol = np.asarray(vol, dtype=np.float32)
    if is_mask:
        return vol > 0
    thr = np.percentile(vol[vol > 0], percentile) if np.any(vol > 0) else 0.0
    return vol > thr
