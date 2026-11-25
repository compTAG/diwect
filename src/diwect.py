"""
This file is a PyTorch implementation of the Directionally Integrated Weighted
Euler Characteristic Transform (DIWECT).
It is based on many coventions established in the pyECT project, see:
https://github.com/compTAG/pyect.

Throughout, we reference Definitions, Theorems, Lemmas, etc. from the paper
An Omnidirectional WECT: Closed-Form Computation of the Directionally-Integrated
Weighted Euler Characteristic Transform.
We also use notation following the paper:
    N(v)            the normalized vector (Definition 11)
    r_v(t)          the geodesic radius of the cap of v at height t (Definition 13)
    c = -N(v)       the center of that cap
    B_r(c)          the closed spherical cap (Definition 12)
    A_sigma(t)      the support of sigma at height t (Definition 8)
    S^n_sigma(v)    the vertex witness region of v in sigma (Definition 9)
    mu              arclength on S^1, area on S^2

This implementation computes the DIWECT (Definition 6, DIWECT)
    DIWECT_K(t) = int_{s in S^n} WECT_K(s, t) ds
exactly (with respect to directions). In other words, no direction sampling
is required.

As input, it accepts weighted simplicial complexes in R^2 or R^3.
This implementation reuses the height grid of pyECT's WECT computation, so that
    DIWECT(num_heights, normalize=True)(complex) ~= WECT(dirs, num_heights)(complex).mean(dim=0)
for uniformly distributed `dirs`.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

# Default chunk sizes calibrated on a T4. You may wish to pass in a chunk_size
# param specific to your own compute environmnet
DEFAULT_CHUNK_SIZE: Dict[int, int] = {2: 1 << 22, 3: 1 << 20}


class DIWECT(torch.nn.Module):
    """
    A torch module for computing the DIWECT of a simplicial complex in R^2 or R^3.

    This module may be used just for computing the DIWECT, or used as a layer in a
    neural network.
    """

    def __init__(
        self,
        num_heights: int,
        normalize: bool = False,
        dtype: torch.dtype = torch.float64,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Initializes the DIWECT module.

        Args:
            num_heights: The number of distinct height values in
                linspace(-M, M, num_heights), where M is the largest vertex norm
                (the same grid used by pyect.WECT).
            normalize: If True, divide by the measure of S^n so the output is the
                average of the WECT over all directions rather than the integral.
            dtype: Floating point type used for the computation and the output.
                float64 is recommended (use float32 on devices without float64,
                such as MPS).
            chunk_size: Maximum number of (simplex, vertex, height) triples
                processed at once to bounds peak memory usage
        """
        super().__init__()
        num_heights = int(num_heights)
        if num_heights <= 0:
            raise ValueError("The num_heights parameter must be positive")
        self.num_heights: int = num_heights
        self.normalize: bool = bool(normalize)
        self.dtype: torch.dtype = dtype
        self.chunk_size: Optional[int] = None if chunk_size is None else int(chunk_size)
        self.eps: float = float(torch.finfo(dtype).eps)

    @torch.jit.export
    def heights(self, v_coords: torch.Tensor) -> torch.Tensor:
        """
        The heights linspace(-M, M, num_heights), with M the largest vertex norm.

        Args:
            v_coords (torch.Tensor): A tensor of shape (k_0, n) of vertex coordinates.

        Returns:
            torch.Tensor: A tensor of shape (num_heights,).
        """
        v_coords = v_coords.to(dtype=self.dtype)
        max_height = torch.linalg.vector_norm(v_coords, dim=1).amax().clamp(min=1e-12)
        steps = torch.arange(self.num_heights, dtype=self.dtype, device=v_coords.device)
        grid = (2.0 * steps - (self.num_heights - 1)) / max(self.num_heights - 1, 1)
        return max_height * grid

    def forward(
        self,
        complex_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Calculates the DIWECT of a complex in R^2 or R^3.

        By Theorem 15 (first equality, from Equations (4) and (6)),
            DIWECT_K(t) = sum_{sigma in K} (-1)^{dim sigma} omega(sigma) mu(A_sigma(t)),
        so this sums the weighted support measures mu(A_sigma(t)) of every
        simplex, computed by _support_measures_2d or _support_measures_3d.

        Args:
            complex_data: A weighted simplicial complex, represented as a list of pairs of tensors.
                complex_data[0] = (v_coords, v_weights):
                    v_coords (torch.Tensor): A tensor of shape (k_0, n) where k_0 is the number of vertices
                    and n is 2 or 3. Rows are the coordinates of the vertices.

                    v_weights (torch.Tensor): A tensor of shape (k_0). Values are the weights of the vertices.

                for i > 0:
                    complex_data[i] = (simp_verts, simp_weights):
                        simp_verts (torch.Tensor): A tensor of shape (k_i, i+1) where k_i is the number of i-simplices.
                        Rows are the vertex sets of the i-simplices.

                        simp_weights (torch.Tensor): A tensor of shape (k_i). Values are the weights of the i-simplices.

        Returns:
            diwect (torch.Tensor): A 1d tensor of shape (self.num_heights,) containing the DIWECT
                evaluated at self.heights(v_coords).
        """
        # extract fixed parameters
        v_coords = complex_data[0][0]
        device = v_coords.device
        v_coords = v_coords.to(dtype=self.dtype)

        # validate dimension is two or three
        n = v_coords.size(1)
        if n != 2 and n != 3:
            raise ValueError("DIWECT is only implemented for complexes in R^2 and R^3")

        # initialize the result tensor
        diwect = torch.zeros(self.num_heights, dtype=self.dtype, device=device)

        # exit trivial case
        if v_coords.size(0) == 0:
            return diwect

        # extract heights from vertex coordinates
        heights = self.heights(v_coords)
        chunk_size = self.chunk_size if self.chunk_size is not None else DEFAULT_CHUNK_SIZE[n]

        # iterate over complex dimensions, transforming to proper device + dtype
        for i in range(len(complex_data)):
            if i == 0:
                simp_verts = torch.arange(v_coords.size(0), device=device).unsqueeze(1)
            else:
                simp_verts = complex_data[i][0].to(device=device, dtype=torch.long)
            simp_weights = complex_data[i][1].to(device=device, dtype=self.dtype)
            if simp_verts.size(0) == 0:
                continue

            # add (-1)^i omega(sigma) mu(A_sigma(t)) (Theorem 15), in chunks
            # of at most chunk_size (simplex, vertex, height) triples
            per_chunk = max(1, chunk_size // (simp_verts.size(1) * heights.size(0)))
            for start in range(0, simp_verts.size(0), per_chunk):
                # (k, i+1, n) vertex coordinates of each i-simplex in the chunk
                pts = v_coords[simp_verts[start:start + per_chunk]]
                if n == 2:
                    measures = _support_measures_2d(pts, heights)
                else:
                    measures = _support_measures_3d(pts, heights, self.eps)
                diwect = diwect + (-1) ** i * (simp_weights[start:start + per_chunk] @ measures)

        if self.normalize:
            diwect = diwect / ((2.0 * math.pi) if n == 2 else (4.0 * math.pi))
        return diwect


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalized(v: torch.Tensor) -> torch.Tensor:
    """The map N: R^{n+1} -> S^n along the last dimension (zero maps to e_0).

    See Definition 11 (Normalized Vector).
    """
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    e0 = torch.zeros_like(v)
    e0[..., 0] = 1.0
    return torch.where(norm > 0, v / torch.where(norm > 0, norm, torch.ones_like(norm)), e0)


def _radius(v_norm: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    """r_v(t) for every vertex norm in v_norm (...) and t in heights (H) -> (..., H).

    See Definition 13 (Geodesic Radius of Spherical Cap)
    """
    v_norm = v_norm.unsqueeze(-1)
    safe_norm = torch.where(v_norm > 0, v_norm, torch.ones_like(v_norm))
    r = torch.acos(torch.clamp(-heights / safe_norm, -1.0, 1.0))
    r_origin = torch.where(
        heights >= 0, torch.full_like(heights, math.pi), torch.zeros_like(heights)
    )
    return torch.where(v_norm > 0, r, r_origin)


# ---------------------------------------------------------------------------
# The DIWECT in two dimensions
# (Section 3.3, Computing the DIWECT in Two Dimensions)
# ---------------------------------------------------------------------------


def _perp(u: torch.Tensor) -> torch.Tensor:
    """u rotated 90 degrees counter-clockwise (see Equation (8))"""
    return torch.stack([-u[..., 1], u[..., 0]], dim=-1)


def _signed_angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """delta(u, v) = atan2(u ^ v, u . v), with range (-pi, pi]

    See Definition 17 (Signed Angle)
    """
    angel = torch.atan2(
        u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(dim=-1)
    )
    
    # clamp delta to (-pi, pi]
    return torch.where(angel <= -math.pi, torch.full_like(angel, math.pi), angel)


def _sgn(x: torch.Tensor) -> torch.Tensor:
    """sgn(x) = 1 if x >= 0, else -1. See Definition 18."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def _support_measures_2d(pts: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    """mu(A_sigma(t)) for simplices pts (k, d+1, 2) and heights (H) -> (k, H)

    See Theorem 15 and Lemma 19.
    """
    d1 = pts.size(1)
    if d1 == 1:
        return 2.0 * _radius(torch.linalg.vector_norm(pts[:, 0], dim=-1), heights)
    if d1 == 3:
        # enforce CCW orientation by swapping v1 and v2 (if needed)
        e1 = pts[:, 1] - pts[:, 0]
        e2 = pts[:, 2] - pts[:, 0]
        cw = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 0
        pts = torch.where(cw[:, None, None], pts[:, [0, 2, 1]], pts)
    elif d1 > 3:
        # enforce CCW orientation
        rel = pts - pts.mean(dim=1, keepdim=True)
        order = torch.atan2(rel[..., 1], rel[..., 0]).argsort(dim=1)
        pts = torch.gather(pts, 1, order.unsqueeze(-1).expand_as(pts))

    # witness arc of vertex v, with (v, next, prev) the vertices in CCW order
    # See Equations (8) and (9)
    nxt = torch.roll(pts, -1, dims=1)
    prev = torch.roll(pts, 1, dims=1)
    u0 = _perp(prev - pts)
    u1 = _perp(pts - nxt)
    center = -_normalized(pts)  # c = -N(v)

    # Lemma 19 with s = c, alpha = delta(u0, c), beta = delta(c, u1)
    alpha = _signed_angle(u0, center).unsqueeze(-1)
    beta = _signed_angle(center, u1).unsqueeze(-1)
    r = _radius(torch.linalg.vector_norm(pts, dim=-1), heights)
    sgn_a, sgn_b = _sgn(alpha), _sgn(beta)
    measure = (
        sgn_a * torch.minimum(r, alpha.abs())
        + sgn_b * torch.minimum(r, beta.abs())
        + 0.5 * r * (1.0 - sgn_a) * (1.0 - sgn_b)
    )
    # sum over v in sigma (Theorem 15)
    return measure.sum(dim=1)


# ---------------------------------------------------------------------------
# The DIWECT in three dimensions
# (Section 3.4, Computing the DIWECT in Three Dimensions)
# ---------------------------------------------------------------------------


def _frames(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors e1, e2 with (e1, e2, a) right-handed orthonormal frames"""
    x_axis = torch.zeros_like(a)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(a)
    y_axis[..., 1] = 1.0
    helper = torch.where(a[..., :1].abs() < 0.9, x_axis, y_axis)
    e1 = _normalized(torch.linalg.cross(a, helper, dim=-1))
    return e1, torch.linalg.cross(a, e1, dim=-1)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    """Angles mapped to [-pi, pi)"""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _cap_region_areas(
    normals: torch.Tensor,
    v_hat: torch.Tensor,
    b: torch.Tensor,
    sign: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Areas of R = B(c) cap S^2_sigma(v) with cap B(c) = {s : c . s >= b}, c = sign * v_hat.

    
    See Lemmas 22, 23, and 26 and Corollary 24. The degenerate cases follow from Lemma 26.

    This implementation was inspired by the geographiclib-rs implementation:
    https://github.com/georust/geographiclib-rs/blob/c5e906d94a4664b5be160eb03f89e76b15cc5ece/src/geodesic.rs#L541.
    However, we use the cap center c as a reference (as opposed to poles/equator),
    and simplify to the spherical case (as opposed to their ellipsoid formulation
    and corrections).

    Args:
        normals: A tensor of shape (k, d+1, d, 3): the unit normals n_j of the
            hemispheres whose intersection is the witness region of each vertex.
        v_hat: A tensor of shape (k, d+1, 3): N(v) for each vertex.
        b: A tensor of shape (k, d+1, H) with values in [0, 1].
        sign: A tensor of shape (H,) with values -1 or 1.
        eps: Machine epsilon of the floating point type.

    Returns:
        torch.Tensor: A tensor of shape (k, d+1, H) with the areas.
    """
    num_hemis = normals.size(2)
    tol = eps ** 0.4  # coincident / tangent circles
    feasible_tol = 100.0 * eps  # round-off only
    pi = math.pi

    # ---- PER (simplex, vertex), independent of height ----
    # great circle j has frame (e1_j, e2_j, n_j) adn v_hat projects onto it with
    # length amp_j = sqrt(1 - (v_hat . n_j)^2) in direction phi_v_j
    # see Lemma 22
    e1, e2 = _frames(normals)
    g1 = (v_hat.unsqueeze(2) * e1).sum(dim=-1)
    g2 = (v_hat.unsqueeze(2) * e2).sum(dim=-1)
    gamma_v = (v_hat.unsqueeze(2) * normals).sum(dim=-1)  # (k, d+1, d)
    amp = torch.hypot(g1, g2)
    phi_v = torch.atan2(g2, g1)

    # hemisphere l restricted to great circle j is the semicircle centered at psi_jl.
    # Relative to the direction of the cap center on circle j,
    # the intersection of these semicircles is the interval [lo, hi]
    l_cos = e1 @ normals.transpose(-1, -2)  # (k, d+1, j, l) = e1_j \cdot n_l
    l_sin = e2 @ normals.transpose(-1, -2)
    psi = torch.atan2(l_sin, l_cos)
    eye = torch.eye(num_hemis, dtype=torch.bool, device=normals.device)
    other = ~eye & (torch.hypot(l_cos, l_sin) > tol)
    shifts = torch.tensor([pi, 0.0], dtype=normals.dtype, device=normals.device)  # sign -1, +1
    rel = _wrap(
        psi.unsqueeze(2) - phi_v.unsqueeze(2).unsqueeze(-1) - shifts.reshape(1, 1, 2, 1, 1)
    )  # (k, d+1, 2, j, l)
    lo_other = torch.where(other.unsqueeze(2), rel - 0.5 * pi, torch.full_like(rel, -math.inf)).amax(dim=-1)
    hi_other = torch.where(other.unsqueeze(2), rel + 0.5 * pi, torch.full_like(rel, math.inf)).amin(dim=-1)

    # for cap circle (f1, sign * f2, c), hemisphere j cuts it at an arc
    # centered at angle p0_j
    # see Lemma 31
    f1, f2 = _frames(v_hat)
    a_j = (normals * f1.unsqueeze(2)).sum(dim=-1)
    b_j = (normals * f2.unsqueeze(2)).sum(dim=-1)
    p0 = torch.stack([torch.atan2(-b_j, a_j), torch.atan2(b_j, a_j)], dim=2)  # (k, d+1, 2, d)

    # ---- PER HEIGHT: gather the sign-dependent data then do teh scalar math ----
    sign_idx = (sign > 0).long()
    lo_other = lo_other.index_select(2, sign_idx)  # (k, d+1, H, d)
    hi_other = hi_other.index_select(2, sign_idx)
    p0 = p0.index_select(2, sign_idx)
    gamma = sign.reshape(1, 1, -1, 1) * gamma_v.unsqueeze(2)  # c \cdot n_j
    bb = b.unsqueeze(-1)
    amp_h = amp.unsqueeze(2)

    # for great-circle arcs, the cap meets circle j in the interval [-h, h] around
    # the cap center's direction (cos(h) = b / amp), intersected with [lo, hi].
    # see Lemmas 22 and 23
    reaches = (amp_h > tol) & (bb < amp_h)
    half = torch.acos(torch.clamp(bb / torch.where(amp_h > tol, amp_h, torch.ones_like(amp_h)), -1.0, 1.0))
    lo = torch.maximum(-half, lo_other)
    hi = torch.minimum(half, hi_other)
    delta = torch.where(reaches, torch.clamp(hi - lo, min=0.0), torch.zeros_like(lo))
    great = 2.0 * torch.atan2(
        torch.sin(0.5 * delta) * gamma,
        torch.cos(0.5 * delta) + amp_h * torch.cos(0.5 * (lo + hi)),
    ).sum(dim=-1)

    # for cap-circle arcs: see Lemmas 29, 30, and 31
    sin_r = torch.sqrt(torch.clamp(1.0 - b * b, min=0.0)).unsqueeze(-1)
    denom = sin_r * amp_h
    has_amp = denom > tol
    ratio = -bb * gamma / torch.where(has_amp, denom, torch.ones_like(denom))
    crosses = has_amp & (ratio.abs() < 1.0)
    cross_half = torch.acos(torch.clamp(ratio, -1.0, 1.0))
    crossings = torch.remainder(torch.stack([p0 - cross_half, p0 + cross_half], dim=-1), 2.0 * pi)
    starts = torch.where(
        crosses.unsqueeze(-1), crossings, torch.full_like(crossings, 2.0 * pi)
    ).flatten(-2)
    starts = torch.sort(starts, dim=-1).values  # (k, d+1, H, 2d)
    ends = torch.cat([starts[..., 1:], starts[..., :1] + 2.0 * pi], dim=-1)
    mids = 0.5 * (starts + ends)
    inside = torch.where(
        has_amp.unsqueeze(-2),
        (ratio < 1.0).unsqueeze(-2)
        & (torch.cos(mids.unsqueeze(-1) - p0.unsqueeze(-2)) >= ratio.unsqueeze(-2) - feasible_tol),
        (bb * gamma >= -feasible_tol).unsqueeze(-2),
    ).all(dim=-1)
    cap_arc = ((ends - starts) * inside).sum(dim=-1)

    # Corollaries 24 and 25
    area = torch.clamp((1.0 - b) * cap_arc + great, 0.0, 4.0 * pi)
    # see Lemma 26
    opposite = ((gamma < 0) & (amp_h <= tol) & (bb < tol)).any(dim=-1)
    return torch.where(opposite, torch.zeros_like(area), area)


def _support_measures_3d(
    pts: torch.Tensor, heights: torch.Tensor, eps: float
) -> torch.Tensor:
    """mu(A_sigma(t)) for simplices pts (k, d+1, 3) and heights (H) -> (k, H)

    See Theorem 15 and Definition 20
    """
    d1 = pts.size(1)
    v_norm = torch.linalg.vector_norm(pts, dim=-1)  # (k, d+1)
    v_hat = _normalized(pts)

    # small-cap reduction: for t <= 0 the cap {v . s <= t} = B_r(-N(v)); for
    # t > 0 use the complement cap {v \cdot s >= t} = B_{pi - r}(N(v)).
    # Both have the form {c \cdot s >= b} with c = sgn(t) N(v) and b = |t| / |v| (<= 1)
    safe_norm = torch.where(v_norm > 0, v_norm, torch.ones_like(v_norm))
    b = torch.clamp(heights.abs() / safe_norm.unsqueeze(-1), max=1.0)  # (k, d+1, H)
    below = heights <= 0
    sign = torch.where(below, -torch.ones_like(heights), torch.ones_like(heights))

    if d1 == 1:
        # vertex case: the witness region is the whole sphere and R is the cap
        # see Lemma 10
        witness = torch.full_like(v_norm, 4.0 * math.pi)
        cap_area = 2.0 * math.pi * (1.0 - b)
    else:
        # otherwise, witness region S^2_sigma(v) = intersection of hemispheres H_{N(v - w)}
        # see Definition 9
        steps = torch.arange(d1, device=pts.device)
        others = (steps.unsqueeze(1) + 1 + steps[: d1 - 1].unsqueeze(0)) % d1 # the indices of the vertices other than i
        normals = _normalized(pts.unsqueeze(2) - pts[:, others])  # (k, d+1, d, 3)
        cap_area = _cap_region_areas(normals, v_hat, b, sign, eps)
        # the two hemispheres {\pm N(v) \cdot s >= 0} split the witness region
        both = torch.tensor([-1.0, 1.0], dtype=pts.dtype, device=pts.device)
        witness = _cap_region_areas(
            normals, v_hat, torch.zeros_like(v_norm).unsqueeze(-1).expand(-1, -1, 2), both, eps
        ).sum(dim=-1)

    # filter to final contribution
    # see lemmas 14 and 26
    cap_area = torch.where(
        (v_norm > 0).unsqueeze(-1) & (b < 1.0 - eps ** 0.4), cap_area, torch.zeros_like(cap_area)
    )
    contribution = torch.where(below, cap_area, witness.unsqueeze(-1) - cap_area)
    at_origin = torch.where(heights >= 0, witness.unsqueeze(-1), torch.zeros_like(cap_area))
    contribution = torch.where((v_norm > 0).unsqueeze(-1), contribution, at_origin)
    return contribution.sum(dim=1)
