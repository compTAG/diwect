from __future__ import annotations

import os
from typing import Tuple

import h5py
import medmnist
import numpy as np
import torch
import torchvision.transforms.functional as TF
from medmnist import INFO

GALAXY10_DECALS_URL = "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
GALAXY10_DECALS_FILE = "Galaxy10_DECals.h5"

# 10 morphology classes (Galaxy10 DECaLS scheme).
GALAXY10_CLASSES = [
    "disturbed", "merging", "round-smooth", "in-between-smooth", "cigar-smooth",
    "barred-spiral", "unbarred-tight-spiral", "unbarred-loose-spiral",
    "edge-on-no-bulge", "edge-on-with-bulge",
]


def load_galaxy10(cache_dir: str) -> Tuple["h5py.Dataset", np.ndarray, "h5py.File"]:  # noqa: F821

    path = os.path.join(cache_dir, GALAXY10_DECALS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it (~2.5 GB) with:\n"
            f"  curl -L -o {path} {GALAXY10_DECALS_URL}"
        )
    f = h5py.File(path, "r")
    images = f["images"]  # lazy (N, 256, 256, 3) uint8
    labels = np.asarray(f["ans"]).astype(np.int64)
    return images, labels, f


def to_wect_image(rgb_uint8: np.ndarray, size: int = 64, bg_percentile: float = 75.0) -> np.ndarray:
    gray = np.asarray(rgb_uint8, dtype=np.float32).mean(axis=2)
    t = torch.from_numpy(gray)[None, None]
    t = TF.resize(t, [size, size], antialias=True)[0, 0].numpy()
    t = t - np.percentile(t, bg_percentile)
    t = np.clip(t, 0.0, None)
    peak = float(t.max())
    if peak > 0:
        t = t / peak
    return t.astype(np.float32)


def stratified_split(labels: np.ndarray, fracs=(0.7, 0.1, 0.2), seed=0):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        train.append(idx[:n_tr])
        val.append(idx[n_tr:n_tr + n_va])
        test.append(idx[n_tr + n_va:])
    return (np.concatenate(train), np.concatenate(val), np.concatenate(test))

_MEDMNIST = {
    "bloodmnist": {"bg_percentile": 0.0},
    "organamnist": {"bg_percentile": 0.0},
}


def get_dataset(name: str, cache_dir: str, wect_size: int = 64) -> dict:
    name = name.lower()
    if name == "galaxy10":
        images, labels, f = load_galaxy10(cache_dir)
        return dict(images=images, labels=labels, split=stratified_split(labels, seed=0),
                    n_classes=10, bg_percentile=75.0, wect_size=wect_size,
                    classes=GALAXY10_CLASSES, close=f.close, name=name)
    if name in _MEDMNIST:
        return _load_medmnist(name, cache_dir, wect_size)
    raise ValueError(f"unknown dataset {name!r}; options: galaxy10, {list(_MEDMNIST)}")


def _load_medmnist(name: str, cache_dir: str, wect_size: int) -> dict:
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    imgs, labels, sizes = [], [], {}
    for split in ["train", "val", "test"]:
        d = DataClass(split=split, download=True, size=64, root=cache_dir)
        a = d.imgs
        if a.ndim == 3:  # grayscale -> RGB
            a = np.repeat(a[..., None], 3, axis=-1)
        imgs.append(a)
        labels.append(np.asarray(d.labels).flatten())
        sizes[split] = len(a)
    images = np.concatenate(imgs).astype(np.uint8)
    labels = np.concatenate(labels).astype(np.int64)
    n_tr, n_va = sizes["train"], sizes["val"]
    split = (np.arange(n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, len(labels)))
    return dict(images=images, labels=labels, split=split, n_classes=len(info["label"]),
                bg_percentile=_MEDMNIST[name]["bg_percentile"], wect_size=wect_size,
                classes=list(info["label"].values()), close=None, name=name)
