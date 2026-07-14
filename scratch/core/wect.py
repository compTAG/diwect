from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from pyect import (
    WECT,
    Complex,
    sample_directions_2d,
    sample_directions_3d,
    weighted_cubical,
    weighted_freudenthal,
)

ArrayLike = Union[np.ndarray, torch.Tensor]
Source = Union[ArrayLike, Complex]

_COMPLEX_BUILDERS = {
    "cubical": weighted_cubical,
    "freudenthal": weighted_freudenthal,
}


def _to_image_tensor(image: ArrayLike, device: Optional[torch.device]) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    elif not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D (h, w) image, got shape {tuple(image.shape)}.")
    return image.to(dtype=torch.float32, device=device)


def sample_directions(
    num_dirs: int, dim: int = 2, device: Optional[torch.device] = None
) -> torch.Tensor:
    if dim == 2:
        return sample_directions_2d(num_dirs, device=device)
    if dim == 3:
        return sample_directions_3d(num_dirs, device=device)
    raise ValueError(f"Only dim in (2, 3) is supported, got dim={dim}.")


def image_to_complex(
    image: ArrayLike,
    complex_type: str = "cubical",
    device: Optional[torch.device] = None,
) -> Complex:
    if complex_type not in _COMPLEX_BUILDERS:
        raise ValueError(
            f"complex_type must be one of {sorted(_COMPLEX_BUILDERS)}, got {complex_type!r}."
        )
    img = _to_image_tensor(image, device)
    return _COMPLEX_BUILDERS[complex_type](img, device=device)


def _as_complex(
    source: Source, complex_type: str, device: Optional[torch.device]
) -> Complex:
    if isinstance(source, Complex):
        return source.to(device) if device is not None else source
    return image_to_complex(source, complex_type=complex_type, device=device)


class WECTTransform:

    def __init__(
        self,
        num_dirs: int = 64,
        num_heights: int = 64,
        dim: int = 2,
        complex_type: str = "cubical",
        dirs: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        if dirs is None:
            dirs = sample_directions(num_dirs, dim=dim, device=self.device)
        else:
            dirs = torch.as_tensor(dirs, dtype=torch.float32, device=self.device)
        self.dirs = dirs
        self.num_dirs = dirs.shape[0]
        self.num_heights = int(num_heights)
        self.dim = dirs.shape[1]
        self.complex_type = complex_type
        self._module = WECT(dirs, self.num_heights)
        if self.device is not None:
            self._module = self._module.to(self.device)

    def wect(self, source: Source, as_numpy: bool = True) -> ArrayLike:
        cplx = _as_complex(source, self.complex_type, self.device)
        with torch.no_grad():
            matrix = self._module(cplx)
        return matrix.cpu().numpy() if as_numpy else matrix

    def diwect(self, source: Source, as_numpy: bool = True) -> ArrayLike:
        matrix = self.wect(source, as_numpy=False)
        return wect_to_diwect(matrix, as_numpy=as_numpy)


def wect_to_diwect(wect_matrix: ArrayLike, as_numpy: bool = True) -> ArrayLike:
    if torch.is_tensor(wect_matrix):
        vec = wect_matrix.mean(dim=0)
        return vec.cpu().numpy() if as_numpy else vec
    arr = np.asarray(wect_matrix)
    return arr.mean(axis=0)


def wect(
    source: Source,
    num_dirs: int = 64,
    num_heights: int = 64,
    dim: int = 2,
    complex_type: str = "cubical",
    dirs: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    as_numpy: bool = True,
) -> ArrayLike:
    transform = WECTTransform(
        num_dirs=num_dirs,
        num_heights=num_heights,
        dim=dim,
        complex_type=complex_type,
        dirs=dirs,
        device=device,
    )
    return transform.wect(source, as_numpy=as_numpy)


def diwect(
    source: Source,
    num_dirs: int = 64,
    num_heights: int = 64,
    dim: int = 2,
    complex_type: str = "cubical",
    dirs: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    as_numpy: bool = True,
) -> ArrayLike:
    transform = WECTTransform(
        num_dirs=num_dirs,
        num_heights=num_heights,
        dim=dim,
        complex_type=complex_type,
        dirs=dirs,
        device=device,
    )
    return transform.diwect(source, as_numpy=as_numpy)
