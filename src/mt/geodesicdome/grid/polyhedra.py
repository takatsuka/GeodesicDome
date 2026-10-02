# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Base polyhedra for geodesic domes, and the nets that lay them out on the index grid.

A geodesic dome is built by cutting every flat face of a base solid into an f-frequency
triangular grid and pushing the grid points out onto the unit sphere.  For fast neighbour
search the solid is first unfolded into a *net* drawn on the same integer grid that
GeodesicDome uses: grid position (x, y) sits at  x * (1, 0) + y * (-1/2, sqrt(3)/2)  in the
plane, so the six neighbours of a point are always the offsets
(+1, 0) (-1, 0) (0, +1) (0, -1) (+1, +1) (-1, -1).

Each net is a :class:`BaseNet`: the corners of the flat base triangles in 3-D, the triangles,
and the grid position of every triangle corner at frequency 1 (frequency f multiplies them).

tetrahedron
    The 4HSOM layout of R. M. de Sousa and R. C. L. Oliveira, "Optimization of geodesic
    self-organizing map using tessellated tetrahedron as spherical lattice", IJCNN 2012:
    the four faces form a parallelogram that the grid stores as an orthogonal array of
    (f + 1) rows by (2f + 1) columns::

        C ---- D ---- C          row f        (C and A appear twice, B and D once)
        |    / |    / |
        |  /   |  /   |
        A ---- B ---- A          row 0

    The left and right columns are the same points, and each of the top and bottom rows is
    folded about its middle point (D and B).  2f^2 + 2 unique points, 4 corners with 3
    neighbours.  D is at the north pole (+y), as in the icosahedral dome.

icosahedron
    The 5-strip net of Wu & Takatsuka (2006).  GeodesicDome keeps its original
    implementation for this base; the net here is the same solid on the generic engine
    and is used to cross-check it.  10f^2 + 2 points, 12 corners with 5 neighbours.

dodecahedron
    Every pentagonal face is cut into 5 triangles meeting at its centre (the pentakis
    dodecahedron, 60 triangles), and each of those into an f-frequency grid, all on the
    flat pentagon.  The 12 pentagon centres have 5 neighbours; the 20 dodecahedron corners
    have 6.  30f^2 + 2 points.  The net is the icosahedral strip net with its 11 cut edges
    replaced by zig-zags along the pentagon edges, so that every triangle stays whole.
    The pentagon centres are in the same directions as the icosahedron's corners.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from mt.geodesicdome import util

__all__ = ['BASES', 'BaseNet', 'normalise_base', 'base_net',
           'tetrahedron_net', 'icosahedron_net', 'dodecahedron_net']

BASES = ('tetrahedron', 'icosahedron', 'dodecahedron')

_ALIASES = {
    'tetrahedron': 'tetrahedron', 'tetra': 'tetrahedron', 'tet': 'tetrahedron', '4': 'tetrahedron',
    'icosahedron': 'icosahedron', 'icosa': 'icosahedron', 'ico': 'icosahedron', '20': 'icosahedron',
    'dodecahedron': 'dodecahedron', 'dodeca': 'dodecahedron', 'dod': 'dodecahedron', '12': 'dodecahedron',
}

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0                      # height of a unit equilateral triangle
_SHEAR = np.array([[1.0, 0.0], [-0.5, _H]])   # grid (x, y) -> plane, rows are the two axes


def normalise_base(base) -> str:
    """
    Canonical name of a base solid: 'tetrahedron', 'icosahedron' or 'dodecahedron'.
    Accepts those names, 'tetra' / 'icosa' / 'dodeca', 'tet' / 'ico' / 'dod' (any case),
    or the number of faces of the solid (4, 20, 12).
    """
    if base is None:
        return 'icosahedron'
    key = str(base).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f'unknown base polyhedron {base!r}; use one of {", ".join(BASES)} '
                         "(or 'tetra', 'icosa', 'dodeca')")
    return _ALIASES[key]


@dataclass(frozen=True)
class BaseNet:
    """
    A base solid unfolded onto the index grid.

        name       'tetrahedron', 'icosahedron' or 'dodecahedron'
        points     (P, 3) corners of the flat base triangles (unit vectors, except the
                   pentagon centres of the dodecahedron, which lie on the flat faces)
        triangles  (T, 3) indices into `points`, counter-clockwise seen from outside
        lattice    (T, 3, 2) integer grid position of each triangle corner at frequency 1;
                   the smallest x and y are 0
        n_faces    number of faces of the solid itself (4, 20, 12)
    """
    name: str
    points: np.ndarray
    triangles: np.ndarray
    lattice: np.ndarray
    n_faces: int

    def grid_size(self, frequency: int) -> tuple[int, int]:
        """(x_max, y_max) of the index grid at the given frequency."""
        top = self.lattice.reshape(-1, 2).max(axis=0) * frequency
        return int(top[0]), int(top[1])

    @property
    def face_centres(self) -> np.ndarray:
        """(n_faces, 3) unit vectors through the centres of the solid's own faces (4, 20 or 12).
        The radial projection of a face is the set of sphere points nearest to its centre."""
        if len(self.triangles) == self.n_faces:                  # triangular faces
            return _unit(self.points[self.triangles].mean(axis=1))
        return _unit(self.points[:self.n_faces])                # dodecahedron: the pentagon centres

    @property
    def points_per_frequency_squared(self) -> int:
        """k in the number of unique points k f^2 + 2."""
        return len(self.triangles) // 2


# ----------------------------------------------------------------------------- helpers
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _to_grid(planar: np.ndarray, b1, b2) -> np.ndarray:
    """Plane coordinates -> integer grid coordinates in the basis (b1, b2); checks integrality."""
    basis = np.array([b1, b2], dtype=float).T            # columns are the basis vectors
    xy = np.linalg.solve(basis, planar.reshape(-1, 2).T).T
    grid = np.rint(xy)
    if not np.allclose(xy, grid, atol=1e-6):
        raise AssertionError('net corner is not on the grid')
    grid = grid.astype(np.int64).reshape(planar.shape)
    return grid - grid.reshape(-1, 2).min(axis=0)


def _signed_area_grid(lattice: np.ndarray) -> np.ndarray:
    q = lattice.astype(float) @ _SHEAR
    a, b = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _outward(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p = points[triangles]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum('ij,ij->i', n, p.sum(axis=1))


def _orientation(points, triangles, lattice) -> int:
    """+1 if every triangle is counter-clockwise on the grid exactly when it faces outward,
    -1 if it is the mirror image, otherwise an error."""
    same = np.sign(_signed_area_grid(lattice)) == np.sign(_outward(points, triangles))
    if same.all():
        return 1
    if not same.any():
        return -1
    raise AssertionError('net is not consistently oriented')


# ----------------------------------------------------------------------------- tetrahedron
def _tetrahedron(sign: float) -> BaseNet:
    colat = np.arccos(-1.0 / 3.0)
    third = 2.0 * np.pi / 3.0
    A = util.spherical_to_xyz(colat, sign * third)
    B = util.spherical_to_xyz(colat, 0.0)
    C = util.spherical_to_xyz(colat, -sign * third)
    D = util.spherical_to_xyz(0.0, 0.0)                  # north pole (+y)
    points = np.array([A, B, C, D])
    a, b, c, d = 0, 1, 2, 3
    # orthogonal array, frequency 1:  row 0 = A B A,  row 1 = C D C
    triangles = np.array([[a, b, d], [a, d, c], [b, a, c], [b, c, d]])
    lattice = np.array([
        [[0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 1], [0, 1]],
        [[1, 0], [2, 0], [2, 1]],
        [[1, 0], [2, 1], [1, 1]],
    ], dtype=np.int64)
    return BaseNet('tetrahedron', points, triangles, lattice, 4)


def tetrahedron_net() -> BaseNet:
    """The 4HSOM orthogonal-array net of the tetrahedron (de Sousa & Oliveira 2012)."""
    net = _tetrahedron(1.0)
    if _orientation(net.points, net.triangles, net.lattice) < 0:
        net = _tetrahedron(-1.0)
    return net


# ----------------------------------------------------------------------------- icosahedron
def _icosahedron_solid(sign: float):
    """12 corners (N, upper ring, lower ring, S), the 20 faces and their positions in the
    5-strip net (plane coordinates, unit edge, x = east, y = north)."""
    colat = np.pi / 2 - np.arctan(0.5)
    step = 2 * np.pi / 5
    N = util.spherical_to_xyz(0.0, 0.0)
    S = util.spherical_to_xyz(np.pi, 0.0)
    U = [util.spherical_to_xyz(colat, sign * step * i) for i in range(5)]
    L = [util.spherical_to_xyz(np.pi - colat, sign * (step * i + step / 2)) for i in range(5)]
    points = np.array([N] + U + L + [S])

    def u(i):
        return 1 + i % 5

    def low(i):
        return 6 + i % 5

    n, s, H = 0, 11, _H
    faces, planar = [], []
    for i in range(5):
        faces += [[u(i), u(i + 1), n], [u(i), low(i), u(i + 1)],
                  [low(i), low(i + 1), u(i + 1)], [low(i), s, low(i + 1)]]
        planar += [[[i, 2 * H], [i + 1, 2 * H], [i + 0.5, 3 * H]],
                   [[i, 2 * H], [i + 0.5, H], [i + 1, 2 * H]],
                   [[i + 0.5, H], [i + 1.5, H], [i + 1, 2 * H]],
                   [[i + 0.5, H], [i + 1, 0.0], [i + 1.5, H]]]
    return points, np.array(faces), np.array(planar, dtype=float)


def _icosahedron_sign() -> float:
    for sign in (1.0, -1.0):
        points, faces, planar = _icosahedron_solid(sign)
        lattice = _to_grid(planar, (1.0, 0.0), (-0.5, _H))
        if _orientation(points, faces, lattice) > 0:
            return sign
    raise AssertionError('cannot orient the icosahedron net')


def icosahedron_net() -> BaseNet:
    """The 5-strip icosahedron net (Wu & Takatsuka 2006) on the generic grid."""
    points, faces, planar = _icosahedron_solid(_icosahedron_sign())
    lattice = _to_grid(planar, (1.0, 0.0), (-0.5, _H))
    return BaseNet('icosahedron', points, faces, lattice, 20)


# ----------------------------------------------------------------------------- dodecahedron
def _reflect(p, a, b):
    """Reflection of point p in the line through a and b."""
    d = (b - a) / np.linalg.norm(b - a)
    v = p - a
    return a + 2 * np.dot(v, d) * d - v


def dodecahedron_net() -> BaseNet:
    """
    The pentakis-dodecahedron net: 60 triangles (pentagon centre + one pentagon edge).

    The pentagon centres are the icosahedron's corners and the dodecahedron's corners are
    the icosahedron's face centres, so the net is drawn over the icosahedron strip net.
    Each icosahedron edge P-Q is crossed by one pentagon edge D1-D2, and the rhombus
    P-D1-Q-D2 holds the two triangles (P, D1, D2) and (Q, D2, D1).  Where the strip net cuts
    P-Q, the rhombus is kept whole on one side (it folds into the empty notch next to it),
    so the cuts run along pentagon edges and grid lines.
    """
    sign = _icosahedron_sign()
    ico, faces, planar = _icosahedron_solid(sign)
    corners = _unit(ico[faces].sum(axis=1))                       # 20 dodecahedron corners
    centres = np.array([corners[(faces == p).any(axis=1)].mean(axis=0) for p in range(12)])
    points = np.vstack([centres, corners])                       # pentagon centres first
    centroid = planar.mean(axis=1)

    where = {}                                                   # edge -> [(face, {vertex: xy})]
    for f, (tri, xy) in enumerate(zip(faces, planar, strict=True)):
        pos = dict(zip(tri.tolist(), xy, strict=True))
        for k in range(3):
            p, q = sorted((int(tri[k]), int(tri[(k + 1) % 3])))
            where.setdefault((p, q), []).append((f, pos))

    triangles, plane = [], []
    for (p, q), ((f1, pos1), (f2, pos2)) in sorted(where.items()):
        pp, pq, c1 = pos1[p], pos1[q], centroid[f1]
        glued = np.allclose(pp, pos2[p]) and np.allclose(pq, pos2[q])
        c2 = centroid[f2] if glued else _reflect(c1, pp, pq)
        for apex, ap, cand in ((p, pp, (12 + f1, c1, 12 + f2, c2)), (q, pq, (12 + f2, c2, 12 + f1, c1))):
            i1, x1, i2, x2 = cand
            tri, xy = [apex, i1, i2], np.array([ap, x1, x2])
            a, b = xy[1] - xy[0], xy[2] - xy[0]
            if a[0] * b[1] - a[1] * b[0] < 0:
                tri, xy = [apex, i2, i1], xy[[0, 2, 1]]
            triangles.append(tri)
            plane.append(xy)

    s = 1.0 / _SQRT3                                             # pentakis triangle side in the plane
    lattice = _to_grid(np.array(plane), (s * _H, s / 2), (-s * _H, s / 2))
    triangles = np.array(triangles)
    if _orientation(points, triangles, lattice) < 0:
        raise AssertionError('dodecahedron net is mirrored')
    return BaseNet('dodecahedron', points, triangles, lattice, 12)


_NETS = {'tetrahedron': tetrahedron_net, 'icosahedron': icosahedron_net, 'dodecahedron': dodecahedron_net}


@cache
def base_net(base) -> BaseNet:
    """The :class:`BaseNet` of a base solid (any name accepted by :func:`normalise_base`)."""
    return _NETS[normalise_base(base)]()
