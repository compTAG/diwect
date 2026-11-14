"""Shared code for the barley seed experiments (notebooks 04a, 04b, 05a, 06).

Data: the founder seeds of Amezquita et al. (2022), Dryad doi:10.5061/dryad.rxwdbrv93,
extracted by ``00-data-barley.ipynb`` into ``data/raw/``.

The pose and ECT conventions are re-implemented from the authors' ``demeter`` code
(https://github.com/amezqui3/demeter): ``misc.rotateSVD`` / ``find_tip`` (pose),
``euler.CubicalComplex`` (cubical complex) and ``directions.pole_directions`` (the
158 directions). The cleaning of the binary scans follows demeter's
``scripts/computing_ecc.py``.
"""
from __future__ import annotations

import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import ndimage
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "barley"
SEED_DIR = RAW / "xray" / "xray_ct_scans" / "clean_founders_seeds"
DESCRIPTORS = RAW / "descriptors" / "shape_descriptors" / "founder_seeds" / "combined_d158_T64.csv"
UMAP_FILE = (RAW / "dimred" / "dimension_reduction" / "umap"
             / "umap_158_16_50_0.1_12_manhattan_unsupervised.csv")

SEED = 0
NUM_THRESHOLDS = 16       # the paper's ECT: 158 directions x 16 thresholds
TRADITIONAL = ["Length", "Width", "Height", "HeightMax", "Shell", "Area", "Vol",
               "ConvexArea", "ConvexVol", "ConvexAreaRatio", "ConvexVolRatio"]


# --------------------------------------------------------------------------- #
# Labels, descriptors and files
# --------------------------------------------------------------------------- #


def load_table(with_files: bool = False) -> dict:
    """The descriptor table of the 3,121 founder seeds.

    Returns accession (the 28 seed types), panicle (the 77 spikes), the 11
    traditional descriptors, and the published ECT (158 directions x 64
    thresholds, and its T = 16 subsample). With ``with_files``, also the path of
    every seed's TIFF scan (which needs the extracted CT scans).
    """
    table = pd.read_csv(DESCRIPTORS)
    published_ect64 = table.iloc[:, 20:].to_numpy(int)
    out = dict(
        table=table,
        accession=table["Founder"].to_numpy(),
        panicle=(table["Scan"] + "-" + table["Color"]).to_numpy(),
        traditional=table[TRADITIONAL].to_numpy(float),
        published_ect64=published_ect64,
        published_ect=published_ect64.reshape(len(table), 158, 64)[:, :, 3::4].reshape(len(table), -1),
    )
    if with_files:
        files = []
        for r in table.itertuples():
            match = glob.glob(f"{SEED_DIR}/{r.Scan}/l{r.Label}_{r.Color}_seeds/{r.Tag}_p*.tif")
            assert len(match) == 1, r.Tag
            files.append(match[0])
        out["files"] = files
    return out


# --------------------------------------------------------------------------- #
# Cleaning, complexes and the paper's pose (from demeter)
# --------------------------------------------------------------------------- #


def pole_directions(parallels=7, meridians=12, x=1, y=0, z=2):
    """demeter's directions: a parallels x meridians grid with poles on axis z (158 directions)."""
    dirs = [np.eye(3)[z]]
    thetas = [(i + 1) * np.pi / (2 * parallels) for i in range(parallels)] + \
             [(i + 1) * np.pi / (2 * parallels) + np.pi / 2 for i in range(parallels - 1)]
    for th in thetas:
        for j in range(meridians):
            ph = j * 2 * np.pi / meridians
            d = np.zeros(3)
            d[x], d[y], d[z] = np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)
            dirs.append(d)
    dirs.append(-np.eye(3)[z])
    dirs = np.array(dirs)
    dirs[np.abs(dirs) < 1e-10] = 0
    return dirs


DIRECTIONS = pole_directions()


def load_seed(path):
    """Binary seed image, cleaned as in demeter's ECT script."""
    import tifffile

    img = tifffile.imread(path) > 0
    img = ndimage.binary_fill_holes(img)
    img = ndimage.binary_closing(img, iterations=2, border_value=0)
    return ndimage.binary_fill_holes(img)


def cubical_complex(img):
    """Voxel-centre cubical complex: vertex coordinates, and edges, squares, cubes as vertex ids."""
    index = -np.ones(np.array(img.shape) + 1, dtype=np.int64)
    coords = np.argwhere(img)
    index[tuple(coords.T)] = np.arange(len(coords))

    def cells(offsets):
        ids = np.stack([index[tuple((coords + o).T)] for o in offsets], 1)
        return ids[(ids >= 0).all(1)]

    unit = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    edges = np.concatenate([cells([(0, 0, 0), o]) for o in unit])
    squares = np.concatenate([cells([(0, 0, 0), a, b, np.add(a, b)])
                              for k, a in enumerate(unit) for b in unit[k + 1:]])
    cubes = cells([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    return coords.astype(float), [edges, squares, cubes]


def paper_frame(coords):
    """demeter's pose: (centroid, R), R's columns the aligned axes (length, width, crease).

    The centred voxel coordinates are rotated by their SVD; the third axis is
    negated when the seed is wider at the bottom than at the top (the crease
    convention). demeter's tip test never fires (the tip is a maximum of centred
    coordinates), so the sign of the length axis is whatever the SVD returns.
    """
    centre = coords.mean(0)
    c = coords - centre
    tip = c[:, 0].max()                           # find_tip(coords, x=2, y=1, z=0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    rot = vh.T.copy()
    s = c @ rot
    pos, neg = s[s[:, 1] > 0], s[s[:, 1] < 0]
    top = pos[pos[:, 2] == pos[:, 2].max()][0, 1] - neg[neg[:, 2] == neg[:, 2].max()][0, 1]
    bottom = pos[pos[:, 2] == pos[:, 2].min()][0, 1] - neg[neg[:, 2] == neg[:, 2].min()][0, 1]
    if bottom > top:
        rot[:, 2] *= -1
    if tip < 0:                                   # never true
        rot[:, 0] *= -1
    return centre, rot


def ect_minmax(verts, cells, dirs, T):
    """The paper's ECT: per-direction Euler curve binned between that direction's min and max height."""
    out = []
    for d in dirs:
        h = verts @ d
        lo, hi = h.min(), h.max()
        ec = np.histogram(h, bins=T, range=(lo, hi))[0].astype(int)
        for k, c in enumerate(cells, 1):
            ec += (-1) ** k * np.histogram(h[c].max(1), bins=T, range=(lo, hi))[0]
        out.append(np.cumsum(ec))
    return np.concatenate(out)


def ect_both_binnings(verts, cells, dirs, T):
    """(min-max ECT, global-grid ECT), each of length len(dirs) * T.

    Min-max: the paper's np.histogram between the direction's min and max.
    Global: pyECT's grid on [-M, M] (M the largest vertex norm), a cell entering
    at the first grid height >= its highest vertex.
    """
    M = np.linalg.norm(verts, axis=1).max()
    minmax, glob_ = [], []
    for d in dirs:
        h = np.sum(verts * d, axis=1)
        top = [h] + [h[c].max(1) for c in cells]
        lo, hi = h.min(), h.max()
        ec = sum((-1) ** k * np.histogram(x, bins=T, range=(lo, hi))[0] for k, x in enumerate(top))
        minmax.append(np.cumsum(ec))
        idx = [np.clip(np.ceil((T - 1) * (M + x) / (2 * M)), 0, T - 1).astype(int) for x in top]
        ec = sum((-1) ** k * np.bincount(i, minlength=T) for k, i in enumerate(idx))
        glob_.append(np.cumsum(ec))
    return np.concatenate(minmax), np.concatenate(glob_)


def freudenthal(img):
    """Freudenthal triangulation of the voxel-centre cubical complex (no new vertices).

    Every cube becomes 6 tetrahedra and every square 2 triangles along its
    monotone diagonal; the result is closed under faces. Returns vertex
    coordinates and [edges, triangles, tetrahedra] as vertex ids.
    """
    coords = np.argwhere(img)
    index = -np.ones(np.array(img.shape) + 1, dtype=np.int64)
    index[tuple(coords.T)] = np.arange(len(coords))
    look = lambda o: index[tuple((coords + o).T)]
    unit = np.eye(3, dtype=int)
    cube = np.all([look(o) >= 0 for o in itertools.product((0, 1), repeat=3)], axis=0)
    tets = np.concatenate([np.stack([look(o)[cube] for o in [(0, 0, 0), unit[p[0]], unit[p[0]] + unit[p[1]], (1, 1, 1)]], 1)
                           for p in itertools.permutations(range(3))])
    tris = []
    for a in range(3):
        for b in range(a + 1, 3):
            square = np.all([look(o) >= 0 for o in [(0, 0, 0), unit[a], unit[b], unit[a] + unit[b]]], axis=0)
            for u, w in [(a, b), (b, a)]:
                tris.append(np.stack([look(o)[square] for o in [(0, 0, 0), unit[u], unit[u] + unit[w]]], 1))
    edges = np.concatenate([np.stack([np.arange(len(coords)), look(unit[a])], 1) for a in range(3)])
    edges = edges[edges[:, 1] >= 0]
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]] + tris)
    faces = np.unique(np.sort(faces, 1), axis=0)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]], edges])
    return coords.astype(float), [np.unique(np.sort(e, 1), axis=0), faces, np.sort(tets, 1)]


def mesh(img, centre):
    """Marching-cubes surface at level 0.5 of the binary image, in voxel units, minus `centre`."""
    from skimage import measure

    verts, faces, *_ = measure.marching_cubes(np.pad(img, 1).astype(np.float32), 0.5)
    verts = verts.astype(np.float64) - 1 - centre             # undo the padding
    return verts, faces.astype(np.int64)


def mesh_edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), 1), axis=0)


def simplicial_complex(verts, faces):
    """A triangle mesh as a unit-weight simplicial complex in pyECT / DIWECT format."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    one = lambda n: torch.ones(n, dtype=torch.float64)
    return [(torch.tensor(verts, dtype=torch.float64), one(len(verts))),
            (torch.tensor(edges), one(len(edges))), (torch.tensor(faces.astype(np.int64)), one(len(faces)))]


# --------------------------------------------------------------------------- #
# Other poses
# --------------------------------------------------------------------------- #


def surface_frame(verts, faces):
    """Area centroid and principal axes (columns, largest variance first) of a triangle
    surface; signs of the first two axes from the third moments, third axis their cross product."""
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    centre = (area[:, None] * tri.mean(1)).sum(0) / area.sum()
    t = tri - centre
    s = t.sum(1)
    second = (np.einsum("f,fki,fkj->ij", area, t, t) + np.einsum("f,fi,fj->ij", area, s, s)) / (12 * area.sum())
    variances, axes = np.linalg.eigh(second)
    variances, axes = variances[::-1], axes[:, ::-1].copy()
    skew = (area[:, None] * (t.mean(1) @ axes[:, :2]) ** 3).sum(0)
    axes[:, :2] *= np.where(skew < 0, -1.0, 1.0)
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return centre, axes, variances


def random_rotation(rng):
    """A uniformly random rotation in SO(3)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def axis_rotation(axis, degrees):
    """Rotation by the given angle about a unit axis (Rodrigues)."""
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


# --------------------------------------------------------------------------- #
# Classification protocol
# --------------------------------------------------------------------------- #

REPEATS = 20


def make_splits(accession, panicle, repeats=REPEATS):
    """The 20 seed splits and 20 spike (panicle) splits used for every feature set.

    * seed split: 25% of every seed type's seeds held out at random;
    * spike split: for every seed type with >= 2 spikes (26 of 28), one whole
      spike held out; the other types are not used.
    """
    multi = [a for a in np.unique(accession) if len(np.unique(panicle[accession == a])) > 1]

    def seed_split(rng):
        test = np.zeros(len(accession), bool)
        for a in np.unique(accession):
            idx = np.flatnonzero(accession == a)
            test[rng.choice(idx, int(round(0.25 * len(idx))), replace=False)] = True
        return ~test, test

    def spike_split(rng):
        test = np.zeros(len(accession), bool)
        for a in multi:
            test |= panicle == rng.choice(np.unique(panicle[accession == a]))
        return np.isin(accession, multi) & ~test, test

    return {"seed": [seed_split(np.random.default_rng([SEED, 5, r])) for r in range(repeats)],
            "spike": [spike_split(np.random.default_rng([SEED, 5, r])) for r in range(repeats)]}


def block(z):
    """Centre, and scale to unit mean squared norm, so blocks of any size weigh the same."""
    z = z - z.mean(0)
    return z / np.sqrt((z ** 2).sum(1).mean())


def _accuracy(x, labels, tr, te, gamma):
    model = SVC(C=10, gamma=gamma).fit(x[tr], labels[tr])
    return np.mean(model.predict(x[te]) == labels[te])


def evaluate(x, labels, splits, gamma="scale", n_jobs=-1):
    """Accuracy of an RBF SVM (C = 10) on every split: {split name: array of 20 accuracies}."""
    return {s: np.array(Parallel(n_jobs=n_jobs)(delayed(_accuracy)(x, labels, tr, te, gamma) for tr, te in sets))
            for s, sets in splits.items()}
