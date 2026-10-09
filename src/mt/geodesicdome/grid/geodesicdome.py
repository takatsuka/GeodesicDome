# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Geodesic domes on the integer index grid of Wu & Takatsuka (2006).

Storage is array-first.  A dome keeps

    _coords   (N, 3)  float64  unit vector of every stored grid position (seam copies included)
    _xy       (N, 2)  int32    grid position (x, y) of every stored position
    _index    (X+1, Y+1) int32 storage row at every grid cell, -1 where nothing is stored
    _cand     (N, 6)  int32    storage row of the six grid neighbours, in _DIRECTIONS order, -1 if absent
    _faces    (F, 3)  int32    triangles as storage rows
    _cls      (N,)    int32    which unique sphere point each stored position is (seam copies share one)

and builds everything with vectorised NumPy.  The storage row of a position is its index in
``get_all_vertices()`` / ``get_all_xyz()`` (columns x = 0, 1, ... and y ascending in a column),
exactly as before.

The :class:`GeodesicVertex` objects that the original API hands out are created lazily, the
first time something asks for them (``vertices``, ``get_all_vertices``, ``get_vertex_at``,
``get_neighbours`` ...).  Code that only needs arrays (``get_all_xyz``, ``get_all_triangles``,
``mt.geodesicdome.compute.DomeArrays``, :meth:`GeodesicDome.within_hops`) never creates them.
"""
from __future__ import annotations

from math import cos, pi
from operator import index as _as_index

import numpy as np

from mt.geodesicdome import util
from mt.geodesicdome.manifold import Manifold
from mt.geodesicdome.vertex import Vertex

from .polyhedra import BASES, BaseNet, base_net, normalise_base

# grid offsets of the six neighbours, in the order get_neighbours visits them
_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (1, 1), (-1, -1), (-1, 0))
_DX = np.array([d[0] for d in _DIRECTIONS], dtype=np.int64)
_DY = np.array([d[1] for d in _DIRECTIONS], dtype=np.int64)


class IGeodesicDome:
    pass


def _normalise_rows(p: np.ndarray) -> np.ndarray:
    """p / |p| along the last axis.  The squared norm goes through matmul, i.e. the same dot
    kernel that ``np.linalg.norm`` uses on a single vector, so results match the per-vertex
    code of earlier versions bit for bit."""
    sq = np.matmul(p[..., None, :], p[..., :, None])[..., 0, 0]
    return p / np.sqrt(sq)[..., None]


def _mean_edge_angle(coords: np.ndarray, faces: np.ndarray) -> float:
    """Mean great-circle length (radians) of the edges of the faces, each face edge counted once."""
    ends = coords[faces[:, [[0, 1], [1, 2], [2, 0]]]].reshape(-1, 2, 3)
    cosine = np.clip(np.einsum('ij,ij->i', ends[:, 0], ends[:, 1]), -1.0, 1.0)
    return float(np.arccos(cosine).mean())


def _partition(a: np.ndarray, b: np.ndarray, n: int, steps: np.ndarray) -> np.ndarray:
    """Points a + j (b - a) / n (j in `steps`) pushed onto the sphere: (E, 3) x (E, 3) -> (E, J, 3)."""
    d = (b - a) / n
    return _normalise_rows(a[:, None, :] + steps[None, :, None] * d[:, None, :])


class GeodesicVertex(Vertex):
    """
    A stored position of a geodesic dome.  ``x``, ``y`` are its index-grid position, ``coord``
    its unit vector (a view of the dome's coordinate array, so writes go to the dome),
    ``same_vertices`` the other stored copies of the same sphere point (``None`` off the seams).
    """

    __slots__ = ('_coord', '_latlon', '_dome', '_i', 'same_vertices', 'frequency', 'projected_coord',
                 'neighbour_mask')

    def __init__(self, latitude=None, longitude=None, coord=None, x=None, y=None, frequency=1):
        self._i = -1
        self._dome = None
        self._coord = None
        self._latlon = None
        super().__init__(x, y)
        self.same_vertices: list[GeodesicVertex] | None = None
        self.frequency = frequency
        if (latitude is not None) and (longitude is not None):
            self._coord = util.spherical_to_xyz(latitude, longitude)
            self._latlon = np.array([latitude, longitude])
        elif coord is not None:
            self._coord = np.asarray(coord, dtype=float)

    @property
    def coord(self) -> np.ndarray:
        c = self._coord
        if c is None and self._dome is not None:
            c = self._coord = self._dome._coords[self._i]      # row view, created on first use
        return c

    @coord.setter
    def coord(self, value) -> None:
        if self._dome is not None:
            self.coord[...] = value                   # write through to the dome's array
        else:
            self._coord = np.asarray(value, dtype=float)
        self._latlon = None

    @property
    def latlon_coord(self) -> np.ndarray:
        """(colatitude from +y, longitude) in radians; computed on first use."""
        ll = self._latlon
        if ll is None:
            ll = self._latlon = util.xyz_to_spherical(self.coord)
        return ll

    @latlon_coord.setter
    def latlon_coord(self, value) -> None:
        self._latlon = value


class GeodesicDome(IGeodesicDome, Manifold):
    """
    A geodesic dome: a base polyhedron whose faces are cut into an f-frequency triangular
    grid and pushed out onto the unit sphere.

        GeodesicDome(8)                          # icosahedron (default): 10 f^2 + 2 points
        GeodesicDome(8, base='tetrahedron')      # 4HSOM tetrahedron:      2 f^2 + 2 points
        GeodesicDome(8, base='dodecahedron')     # pentakis dodecahedron: 30 f^2 + 2 points

    ``base`` accepts 'tetrahedron' / 'tetra', 'icosahedron' / 'icosa', 'dodecahedron' /
    'dodeca' (see :func:`mt.geodesicdome.grid.polyhedra.normalise_base`).  Calling
    ``GeodesicDome(...)`` returns an :class:`IcosahedronDome`, :class:`TetrahedronDome` or
    :class:`DodecahedronDome`; all of them are GeodesicDomes with the same interface:
    vertices on an integer index grid (``vertex.x``, ``vertex.y``) whose six neighbours are
    the offsets (+-1, 0), (0, +-1), (+1, +1), (-1, -1), seam copies in
    ``vertex.same_vertices``, ``get_vertex_at``, ``get_neighbours``,
    ``get_neighbours_in_distance``, ``get_faces``, ``split``.

    Array API (no vertex objects are created):

        dome.get_all_xyz()                 (N, 3) coordinates, row = storage index = vertex.id
        dome.get_all_triangles()           (3F,) storage indices of the faces
        dome.neighbour_ids(i)              storage indices of the neighbours of position i
        dome.within_hops(i, hops)          positions within `hops` grid steps (one per sphere point)
        dome.within_arc(i, angle)          positions within a great-circle angle (radians)

    Lloyd relaxation (moves the points, keeps the grid; see :mod:`mt.geodesicdome.relax`):

        GeodesicDome(16, relax=True)       built and relaxed to convergence
        GeodesicDome(16, relax={'iters': 100, 'omega': 1.0})   with other settings
        dome.relax()                       relax an existing dome in place
    """

    base: str = 'icosahedron'

    def __new__(cls, frequency=1, base=None, *args, **kwargs):
        if cls is GeodesicDome:
            cls = _DOME_CLASSES[normalise_base(base)]
        return super().__new__(cls)

    def _check_base(self, base) -> None:
        if base is not None and normalise_base(base) != self.base:
            raise ValueError(f'{type(self).__name__} is built on the {self.base}, not on {base!r}')

    @staticmethod
    def _check_factor(frequency) -> int:
        f = _as_index(frequency) if not isinstance(frequency, float) else int(frequency)
        if f < 1 or f != frequency:
            raise ValueError('frequency must be a positive integer')
        return f

    # ------------------------------------------------------------------ array state
    def _set_arrays(self, xy: np.ndarray, index: np.ndarray, coords: np.ndarray, faces: np.ndarray,
                    cand: np.ndarray, same: dict[int, list[int]], cls: np.ndarray) -> None:
        self._xy = xy
        self._index = index
        self._coords = coords
        self._faces = faces
        self._cand = cand
        self._same = same
        self._cls = cls
        self._n_points = int(cls.max()) + 1 if len(cls) else 0
        old = getattr(self, '_flat', None)
        if old is not None:              # rebuilt by split(): old vertices keep their old coordinates
            for v in old:
                v.coord             # noqa: B018  (materialise the view of the old array)
                v._dome = None
        # lazily built
        self._flat = None
        self._columns = None
        self._ring = None
        self._bfs = None
        self.triangles = []
        self.relaxed = False             # coordinates are fresh subdivisions until relax() moves them
        self.relax_steps = 0

    @staticmethod
    def _neighbour_candidates(xy: np.ndarray, index: np.ndarray) -> np.ndarray:
        """(N, 6) storage index of the six grid neighbours of every position (-1 if none stored)."""
        nx = xy[:, :1] + _DX[None, :]
        ny = xy[:, 1:] + _DY[None, :]
        inside = (nx >= 0) & (ny >= 0) & (nx < index.shape[0]) & (ny < index.shape[1])
        cand = index[np.where(inside, nx, 0), np.where(inside, ny, 0)]
        cand[~inside] = -1
        return cand.astype(np.int32, copy=False)

    # ------------------------------------------------------------------ vertex objects (lazy)
    def _objects(self) -> list[GeodesicVertex]:
        flat = self._flat
        if flat is None:
            flat = self._materialise()
        return flat

    def _materialise(self) -> list[GeodesicVertex]:
        n = len(self._coords)
        xs = self._xy[:, 0].tolist()
        ys = self._xy[:, 1].tolist()
        f = self.frequency
        new, gv = object.__new__, GeodesicVertex
        flat = [None] * n
        for i in range(n):
            v = new(gv)
            v.visited = False
            v.x = xs[i]
            v.y = ys[i]
            v.id = i
            v._i = i
            v._coord = None                             # view of _coords[i], made on first use
            v._latlon = None
            v._dome = self
            v.same_vertices = None
            v.frequency = f
            v.data = None
            v.color = None
            v.manifold = self
            flat[i] = v
        for i, others in self._same.items():
            flat[i].same_vertices = [flat[j] for j in others]
        self._flat = flat
        self._after_materialise(flat)
        return flat

    def _after_materialise(self, flat) -> None:
        pass

    @property
    def vertices(self) -> list[list[GeodesicVertex]]:
        """The stored vertices column by column: ``vertices[x]`` is column x, y ascending."""
        cols = self._columns
        if cols is None:
            flat = self._objects()
            bounds = np.searchsorted(self._xy[:, 0], np.arange(self.x_max + 2)).tolist()
            cols = self._columns = [flat[bounds[x]:bounds[x + 1]] for x in range(self.x_max + 1)]
        return cols

    def get_all_vertices(self) -> list[GeodesicVertex]:
        return list(self._objects())

    def get_number_of_vertices_per_face(self) -> int:
        return 3

    def get_vertex_at(self, x, y) -> GeodesicVertex | None:
        if 0 <= x <= self.x_max and 0 <= y <= self.y_max:
            i = int(self._index[x, y])
            if i >= 0:
                return self._objects()[i]
        return None

    def _updateIDs(self) -> None:
        if self._flat is not None:
            for i, v in enumerate(self._flat):
                v.id = i

    def _build_faces(self) -> list[GeodesicVertex]:
        self._updateIDs()
        flat = self._objects()
        self.triangles = [flat[i] for i in self._faces.ravel().tolist()]
        return self.triangles

    def get_faces(self) -> list[GeodesicVertex]:
        return self._build_faces()

    def get_all_xyz(self) -> np.ndarray:
        return self._coords.copy()

    def get_all_triangles(self) -> np.ndarray:
        self._updateIDs()
        return self._faces.ravel().astype(np.int64)

    def unmark_vertices(self) -> None:
        flat = self._flat
        if flat is not None:
            for v in flat:
                v.visited = False

    _unmark_vertices = unmark_vertices

    # ------------------------------------------------------------------ neighbour search on vertices
    def _neighbour_table(self) -> list[list[GeodesicVertex]]:
        """For every position, its stored grid neighbours as vertex objects, in _DIRECTIONS order."""
        flat = self._objects()
        cand = self._cand
        objs = np.empty(len(flat) + 1, dtype=object)
        objs[:-1] = flat
        missing = cand < 0
        order = np.argsort(missing, axis=1, kind='stable')         # present ones first, in order
        table = objs[np.take_along_axis(cand, order, axis=1)].tolist()
        counts = 6 - missing.sum(axis=1)
        for i in np.flatnonzero(counts < 6).tolist():
            del table[i][counts[i]:]
        table = list(map(tuple, table))
        self._ring = table
        return table

    def _row_of(self, v: GeodesicVertex) -> int:
        if getattr(v, '_dome', None) is self:
            return v._i
        i = self._index[v.x, v.y] if (0 <= v.x <= self.x_max and 0 <= v.y <= self.y_max) else -1
        if i < 0:
            raise ValueError(f'no vertex of this dome at grid position ({v.x}, {v.y})')
        return int(i)

    def _expand(self, frontier, table) -> list[GeodesicVertex]:
        """Unvisited neighbours (seam copies included) of every vertex in `frontier`, marking
        them -- exactly what calling get_neighbours(u, False) on each u in turn returns."""
        out = []
        add = out.append
        for u in frontier:
            for n in table[u._i]:
                if not n.visited:
                    n.visited = True
                    add(n)
                    s = n.same_vertices
                    if s:
                        for t in s:
                            t.visited = True
            same = u.same_vertices
            if same:
                for s in same:
                    for n in table[s._i]:
                        if not n.visited:
                            n.visited = True
                            add(n)
                            t2 = n.same_vertices
                            if t2:
                                for t in t2:
                                    t.visited = True
        return out

    def _start(self, v: GeodesicVertex) -> GeodesicVertex:
        i = self._row_of(v)
        u = self._objects()[i]
        v.visited = True
        u.visited = True
        if u.same_vertices:
            for s in u.same_vertices:
                s.visited = True
        return u

    def get_neighbours(self, v: GeodesicVertex, visit_same_vertex: bool = False) -> list[GeodesicVertex]:
        """
        The not-yet-visited direct neighbours of `v` (and, unless `visit_same_vertex`, of its
        seam copies), marking them and their copies as visited.  Call ``unmark_vertices()``
        before a fresh search.
        """
        table = self._ring if self._ring is not None else self._neighbour_table()
        u = self._start(v)
        if visit_same_vertex or not u.same_vertices:
            out = []
            for n in table[u._i]:
                if not n.visited:
                    n.visited = True
                    out.append(n)
                    s = n.same_vertices
                    if s:
                        for t in s:
                            t.visited = True
            return out
        return self._expand((u,), table)

    def get_neighbours_in_distance(self, v: GeodesicVertex, dis: int) -> list[list[GeodesicVertex]]:
        """Rings 1..dis around `v`: ``result[k]`` holds the vertices k + 1 grid steps away."""
        table = self._ring if self._ring is not None else self._neighbour_table()
        ring = self._expand((self._start(v),), table)
        rings = [ring]
        for _ in range(dis - 1):
            ring = self._expand(ring, table)
            rings.append(ring)
        return rings

    # ------------------------------------------------------------------ array (index) queries
    def _check_row(self, i) -> int:
        i = _as_index(i)
        if not 0 <= i < len(self._coords):
            raise IndexError('grid position index out of range')
        return i

    def _bfs_state(self):
        """
        The graph of distinct sphere points, each represented by its first stored position:
        adj[p] is a tuple of representative positions for every representative p (None for
        other seam copies), rep_of[p] the representative of position p, marks a visit stamp
        per position.  Int objects are shared, so this costs about one tuple per point.
        """
        bfs = self._bfs
        if bfs is None:
            cand, cls = self._cand, self._cls
            n = len(cls)
            rep = np.full(self._n_points, n, dtype=np.int64)
            np.minimum.at(rep, cls, np.arange(n))
            rep_of = rep[cls]
            ok = cand.ravel() >= 0
            src = np.repeat(rep_of, 6)
            dst = rep_of[np.where(ok, cand.ravel(), 0)]
            ok &= src != dst
            key = np.unique(src[ok] * n + dst[ok])
            s, d = key // n, key % n
            owners, starts, counts = np.unique(s, return_index=True, return_counts=True)
            width = int(counts.max()) if len(counts) else 0
            padded = np.zeros((len(owners), width), dtype=np.int64)
            row = np.repeat(np.arange(len(owners)), counts)
            padded[row, np.arange(len(s)) - starts[row]] = d
            pool = np.arange(n).astype(object)               # one int object per position, shared
            rows = pool[padded].tolist()
            for r in np.flatnonzero(counts < width).tolist():
                del rows[r][counts[r]:]
            adj = [None] * n
            for p, r in zip(owners.tolist(), rows, strict=True):
                adj[p] = tuple(r)
            bfs = self._bfs = [adj, pool[rep_of].tolist(), [0] * n, 0]
        return bfs

    def neighbour_ids(self, i: int) -> list[int]:
        """Storage indices of the direct neighbours of position `i` (one per sphere point, seams included)."""
        return self.within_hops(i, 1, include_self=False)

    def within_hops(self, i: int, hops: int, include_self: bool = True) -> list[int]:
        """
        Storage indices of the positions at most `hops` grid steps from position `i`, in
        breadth-first order, one per sphere point (a seam point is reported by its first
        stored copy).  No vertex objects are created and no visited flags are touched.
        """
        i = self._check_row(i)
        hops = _as_index(hops)
        if hops < 0:
            raise ValueError('hops must be a non-negative integer')
        bfs = self._bfs if self._bfs is not None else self._bfs_state()
        adj, marks = bfs[0], bfs[2]
        epoch = bfs[3] = bfs[3] + 1
        if epoch >= 1 << 62:
            marks[:] = [0] * len(marks)
            epoch = bfs[3] = 1
        r0 = bfs[1][i]
        marks[r0] = epoch
        out = [i] if include_self else []
        frontier = [r0]
        for _ in range(hops):
            nxt = []
            add = nxt.append
            for c in frontier:
                for d in adj[c]:
                    if marks[d] != epoch:
                        marks[d] = epoch
                        add(d)
            if not nxt:
                break
            out += nxt
            frontier = nxt
        return out

    def within_arc(self, i: int, angle: float, include_self: bool = True) -> np.ndarray:
        """Storage indices of the positions within great-circle `angle` (radians) of position `i`
        (seam copies included).  For many queries at once see :mod:`mt.geodesicdome.compute`."""
        i = self._check_row(i)
        if not 0.0 <= angle <= pi:
            raise ValueError('angle must be between 0 and pi radians')
        mask = self._coords @ self._coords[i] >= cos(angle) - 1e-14
        if not include_self:
            mask[self._cls == self._cls[i]] = False
        return np.flatnonzero(mask)

    @property
    def n_points(self) -> int:
        """Number of distinct sphere points (seam copies counted once)."""
        return self._n_points

    @property
    def point_index(self) -> np.ndarray:
        """(N,) the distinct sphere point stored at each position; seam copies share a value."""
        return self._cls.copy()

    # ------------------------------------------------------------------ Lloyd relaxation
    def relax(self, iters: int | None = None, omega: float | None = None, tol: float | None = None,
              backend=None) -> GeodesicDome:
        """
        Moves every point towards the centroid of its cell (Lloyd relaxation on the dome's own
        triangles, see :mod:`mt.geodesicdome.relax`), evening out the cell areas.  The grid is
        untouched: every (x, y) position, neighbour, face and seam copy stays as it was, so all
        index and neighbour queries give the same results; only the coordinates change.
        Seam copies keep identical coordinates.  Coordinates are changed in place, so existing
        vertex objects see the new ones.

        The defaults (``mt.geodesicdome.relax.CONVERGED``: up to 20 000 steps, ``omega = 1.8``,
        stop when no point moves 1e-7 rad) run to convergence.  ``iters=100, omega=1.0`` is plain
        Lloyd with a fixed number of steps.

        The steps run on the compute backend (:mod:`mt.geodesicdome.backend`): a CUDA or Apple
        GPU when one is available and the dome has at least
        ``mt.geodesicdome.relax.GPU_MIN_POINTS`` points, otherwise NumPy on the CPU.  On the
        Apple GPU (float32 only) the last few steps are made on the CPU in float64, so the
        result is as precise as a CPU run.

        A later :meth:`split` subdivides the relaxed coordinates and returns an unrelaxed dome
        (``relaxed`` is reset); call ``relax()`` again after it.

        :param iters: maximum number of steps
        :param omega: over-relaxation factor (1 = plain Lloyd; keep it below 2)
        :param tol: stop once no point moves more than `tol` radians in a step
        :param backend: None (automatic, see above), or 'cpu', 'cuda', 'mps', 'torch', 'cupy' or a Backend
        :return: the dome itself; ``dome.relaxed`` is True and ``dome.relax_steps`` the number of steps taken
        """
        from mt.geodesicdome import relax as lloyd_relax

        kw = dict(lloyd_relax.CONVERGED)
        for k, v in (('iters', iters), ('omega', omega), ('tol', tol)):
            if v is not None:
                kw[k] = v
        if kw['iters'] < 0:
            raise ValueError('iters must be non-negative')
        if not 0.0 < kw['omega'] < 2.0:
            raise ValueError('omega must be between 0 and 2')
        cls = self._cls
        x = np.empty((self._n_points, 3), dtype=np.float64)
        x[cls] = self._coords
        tri = lloyd_relax.outward(x, cls[self._faces])
        y, steps = lloyd_relax.lloyd(x, tri, return_steps=True, backend=backend, **kw)
        self._coords[...] = y[cls]
        if self._flat is not None:
            for v in self._flat:
                v._latlon = None
        self.relaxed = True
        self.relax_steps = self.relax_steps + steps
        self._after_relax()
        return self

    def _after_relax(self) -> None:
        pass

    def _apply_relax_option(self, relax) -> None:
        """The constructor's ``relax`` option: False, True (converged defaults) or a dict of relax() arguments."""
        if relax is None or relax is False:
            return
        if relax is True:
            self.relax()
        elif isinstance(relax, dict):
            self.relax(**relax)
        else:
            raise TypeError('relax must be True, False or a dict of relax() arguments (iters, omega, tol, backend)')

    def _relaxed_repr(self) -> str:
        return ', relaxed=True' if self.relaxed else ''


class IcosahedronDome(GeodesicDome):
    """
    A Geodesicdome based on the Icosahedron (22 vertices and 20 triangles)
    The base Icosahedron's vertices are arranged on a rectilinear grid as shown below.
    This is to use an indexing scheme to quickly search vertex's i-level neighbours.

    #                  v17  v20  v22
    #             v13  v16  v19  v21
    #         v9  v12  v15  v18
    #     v5  v8  v11  v14
    # v2  v4  v7  v10
    # v1  v3  v6
    """

    """
    Icosahedron's approximated average arc length
    """
    _arc_length = 1.106588

    base = 'icosahedron'

    def __init__(self, frequency=1, base=None, relax=False):
        super().__init__()
        self._check_base(base)
        frequency = self._check_factor(frequency)
        self.arcLength = IcosahedronDome._arc_length  # approximate the average arc length
        self.frequency = 1
        self.x_max = 6
        self.y_max = 5

        d_az = 2 * np.pi / 5
        d_lat = ((90 - 26.565) / 180) * np.pi
        # (x, y, colatitude, longitude) of v1 .. v22
        spec = [(0, 0, d_lat, 0), (0, 1, 0, 0),
                (1, 0, np.pi - d_lat, d_az * 0.5), (1, 1, d_lat, d_az), (1, 2, 0, 0),
                (2, 0, np.pi, 0), (2, 1, np.pi - d_lat, d_az * 1.5), (2, 2, d_lat, d_az * 2), (2, 3, 0, 0),
                (3, 1, np.pi, 0), (3, 2, np.pi - d_lat, d_az * 2.5), (3, 3, d_lat, d_az * 3), (3, 4, 0, 0),
                (4, 2, np.pi, 0), (4, 3, np.pi - d_lat, d_az * 3.5), (4, 4, d_lat, d_az * 4), (4, 5, 0, 0),
                (5, 3, np.pi, 0), (5, 4, np.pi - d_lat, d_az * 4.5), (5, 5, d_lat, 0),
                (6, 4, np.pi, 0), (6, 5, np.pi - d_lat, d_az * 0.5)]
        xy = np.array([(x, y) for x, y, _, _ in spec], dtype=np.int32)
        coords = np.array([util.spherical_to_xyz(t, lon) for _, _, t, lon in spec], dtype=np.float64)
        top = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]          # v2 v5 v9 v13 v17 (north pole)
        bottom = [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)]       # v6 v10 v14 v18 v21 (south pole)
        same = {(0, 0): [(5, 5)], (5, 5): [(0, 0)], (1, 0): [(6, 5)], (6, 5): [(1, 0)]}
        for group in (top, bottom):
            for p in group:
                same[p] = [q for q in group if q != p]
        self._finish(xy, coords, same)
        if frequency > 1:
            self.split(frequency)
        self._apply_relax_option(relax)

    def __repr__(self) -> str:
        return f'GeodesicDome(frequency={self.frequency}{self._relaxed_repr()})'

    # ------------------------------------------------------------------ construction
    def _finish(self, xy: np.ndarray, coords: np.ndarray, same_xy: dict) -> None:
        """Index grid, faces, neighbours and seam classes from the stored positions."""
        index = np.full((self.x_max + 1, self.y_max + 1), -1, dtype=np.int32)
        index[xy[:, 0], xy[:, 1]] = np.arange(len(xy), dtype=np.int32)
        valid = index >= 0
        # faces: every grid cell whose four corners are stored, cut along (x, y)-(x+1, y+1)
        cx, cy = np.nonzero(valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:])
        o, r, d, u = index[cx, cy], index[cx + 1, cy], index[cx + 1, cy + 1], index[cx, cy + 1]
        faces = np.stack([np.stack([o, r, d], axis=1), np.stack([o, d, u], axis=1)], axis=1).reshape(-1, 3)
        same = {int(index[p]): [int(index[q]) for q in qs] for p, qs in same_xy.items()}
        # seam classes: union-find over the (few) seam positions
        parent = {}

        def find(a):
            while parent.get(a, a) != a:
                parent[a] = parent.get(parent[a], parent[a])
                a = parent[a]
            return a

        for p, qs in same.items():
            for q in qs:
                rp, rq = find(p), find(q)
                if rp != rq:
                    parent[max(rp, rq)] = min(rp, rq)
        root = np.arange(len(xy), dtype=np.int64)
        if same:
            keys = np.fromiter(same.keys(), dtype=np.int64, count=len(same))
            root[keys] = [find(int(k)) for k in keys]
        _, cls = np.unique(root, return_inverse=True)
        self._same_xy = same_xy
        self._set_arrays(xy, index, coords, faces, self._neighbour_candidates(xy, index), same,
                         cls.reshape(-1).astype(np.int32))

    def _find_same_vertices(self, same: dict, column_range) -> None:
        """
        After a split: pairs up the new positions on the cut edges of the net, walking the cuts
        exactly as the original implementation did (positions are (x, y) tuples).
        """
        def has(p):
            return bool(same.get(p))

        lo0, hi0 = column_range(0)
        top_current = (0, hi0)
        top_next = None
        # stage1, travel through the top vertices that has 4 same vertices points (v2 v5 v9 v13 v17)
        i = 0
        while i < len(same[top_current]):
            top_next = same[top_current][i]
            x1, y1 = top_current
            x2, y2 = top_next
            top_middle = (x1 + 1, y1)
            while (x1 + 1) != x2:
                if not has(top_middle):
                    match = (x2, y2 - 1)
                    same[top_middle] = [match]
                    same[match] = [top_middle]
                x1 += 1
                y2 -= 1
                top_middle = (x1 + 1, y1)
            top_current = top_next
            i += 1

        # stage 2, travel through the flat top vertex (v20, v21)
        v17_x, v17_y = top_next
        v2_y = hi0
        for x in range(v17_x + 1, self.x_max):
            high = (x, v17_y)
            if has(high):
                continue
            xdiff = x - v17_x
            low = (0, v2_y - xdiff) if xdiff < v2_y else (xdiff - v2_y, 0)
            same[high] = [low]
            same[low] = [high]

        # through v21 to v22
        lo, hi = column_range(self.x_max)
        last = [(self.x_max, y) for y in range(lo, hi + 1)]
        v22 = last[-1]
        v3 = same[v22][0]
        for k in range(len(last) - 2, 0, -1):
            test = last[k]
            if has(test):
                continue
            match = (v3[0] + (v22[1] - test[1]), 0)
            same[match] = [test]
            same[test] = [match]

        # stage 3, travel through the bottom vertices that has 4 same vertices (v21 v18 v14 v10 v6)
        v1 = last[0]
        i = len(same[v1]) - 1
        while i >= 0:
            v2 = same[v1][i]
            x1, y1 = v1
            x2, y2 = v2
            v3 = (x1 - 1, y1)
            while (x1 - 1) != x2:
                if not has(v3):
                    v4 = (x2, y2 + 1)
                    same[v3] = [v4]
                    same[v4] = [v3]
                x1 -= 1
                y2 += 1
                v3 = (x1 - 1, y1)
            v1 = v2
            i -= 1

    def split(self, frequency):
        """
        Subdivides every edge of the current dome into `frequency` segments.
        Calls are cumulative: split(2) followed by split(3) gives frequency 6.
        The vertices are rebuilt, so vertex objects from before the split are not reused.

        :param frequency: subdivision factor (an int >= 1)
        :return: None
        """
        f = self._check_factor(frequency)
        if f == 1:
            return
        old_index, old_coords = self._index, self._coords
        valid = old_index >= 0
        X, Y = self.x_max * f, self.y_max * f

        # which new grid cells are stored: old vertices, points on old column / row edges,
        # and every point of an old cell
        vert = valid[:, :-1] & valid[:, 1:]                       # (x, y)-(x, y+1)
        hori = valid[:-1, :] & valid[1:, :]                       # (x, y)-(x+1, y)
        cell = vert[:-1, :] & vert[1:, :] & hori[:, :-1] & hori[:, 1:]
        vx, vy = np.nonzero(vert)
        hx, hy = np.nonzero(hori)
        cx, cy = np.nonzero(cell)
        ox, oy = np.nonzero(valid)
        steps = np.arange(1, f)
        full = np.arange(f + 1)
        new_valid = np.zeros((X + 1, Y + 1), dtype=bool)
        new_valid[f * ox, f * oy] = True
        new_valid[f * vx[:, None], f * vy[:, None] + steps] = True
        new_valid[f * hx[:, None] + steps, f * hy[:, None]] = True
        new_valid[f * cx[:, None, None] + full[None, :, None], f * cy[:, None, None] + full[None, None, :]] = True

        nx, ny = np.nonzero(new_valid)
        index = np.full((X + 1, Y + 1), -1, dtype=np.int32)
        index[nx, ny] = np.arange(len(nx), dtype=np.int32)
        coords = np.empty((len(nx), 3), dtype=np.float64)

        def put(x, y, p):
            coords[index[x, y]] = p

        def get(x, y):
            return coords[index[x, y]]

        # old vertices keep their coordinates
        put(f * ox, f * oy, old_coords[old_index[ox, oy]])
        if f > 1:
            # columns, bottom to top; rows, left to right; cell diagonals (x, y) -> (x+1, y+1)
            put(f * vx[:, None], f * vy[:, None] + steps,
                _partition(old_coords[old_index[vx, vy]], old_coords[old_index[vx, vy + 1]], f, steps))
            put(f * hx[:, None] + steps, f * hy[:, None],
                _partition(old_coords[old_index[hx, hy]], old_coords[old_index[hx + 1, hy]], f, steps))
            put(f * cx[:, None] + steps, f * cy[:, None] + steps,
                _partition(old_coords[old_index[cx, cy]], old_coords[old_index[cx + 1, cy + 1]], f, steps))
            # cell interiors, on lines parallel to the diagonal:
            #   upper-left triangle (O, D, U): from the left column up to the top row
            #   lower-right triangle (O, R, D): from the bottom row up to the right column
            bx, by = f * cx, f * cy
            for h in range(2, f):
                k = np.arange(1, h)
                put(bx[:, None] + k, by[:, None] + (f - h) + k,
                    _partition(get(bx, by + f - h), get(bx + h, by + f), h, k))
                put(bx[:, None] + (f - h) + k, by[:, None] + k,
                    _partition(get(bx + f - h, by), get(bx + f, by + h), h, k))

        self.frequency *= f
        self.x_max, self.y_max = X, Y
        self.arcLength /= f
        same = {(f * x, f * y): [(f * a, f * b) for a, b in qs] for (x, y), qs in self._same_xy.items()}
        first = np.searchsorted(nx, np.arange(X + 2))

        def column_range(x):
            return int(ny[first[x]]), int(ny[first[x + 1] - 1])

        self._find_same_vertices(same, column_range)
        self._finish(np.stack([nx, ny], axis=1).astype(np.int32), coords, same)


class NetDome(GeodesicDome):
    """
    A geodesic dome laid out on the index grid by a :class:`~mt.geodesicdome.grid.polyhedra.BaseNet`.

    Every base triangle of the net is cut into an f-frequency grid; each grid point is
    interpolated on the flat face and projected onto the unit sphere.  A point that the net
    shows more than once (on a cut) is stored once per position, and the copies list each
    other in ``same_vertices``, exactly as in the icosahedral dome.

    Neighbour search is pure index arithmetic: a vertex at (x, y) looks at the six offsets
    (+-1, 0), (0, +-1), (+1, +1), (-1, -1) in the grid, keeping those whose edge lies inside
    the net (a 6-bit mask per vertex, set at construction), and then continues from its seam
    copies.
    """

    base: str = ''

    def __init__(self, frequency=1, base=None, relax=False):
        super().__init__()
        self._check_base(base)
        frequency = self._check_factor(frequency)
        self.net: BaseNet = base_net(self.base)
        self.frequency = 1
        self._build(frequency)
        self._apply_relax_option(relax)

    def __repr__(self) -> str:
        return f'GeodesicDome(frequency={self.frequency}, base={self.base!r}{self._relaxed_repr()})'

    def _after_relax(self) -> None:
        self.arcLength = _mean_edge_angle(self._coords, self._faces)

    # ------------------------------------------------------------------ construction
    def split(self, frequency):
        """
        Subdivides every edge of the current dome into `frequency` segments.
        Calls are cumulative: split(2) followed by split(3) gives frequency 6.
        The vertices are rebuilt, so vertex objects from before the split are not reused.

        :param frequency: subdivision factor (an int >= 1)
        :return: None
        """
        f = self._check_factor(frequency)
        if f > 1:
            self._build(self.frequency * f)

    @staticmethod
    def _template(f: int):
        """Local lattice of one f-frequency triangle: point (i, j) for i + j <= f in the
        original loop order, and its small triangles in the original order."""
        ii = np.concatenate([np.full(f + 1 - i, i) for i in range(f + 1)])
        jj = np.concatenate([np.arange(f + 1 - i) for i in range(f + 1)])
        local = np.full((f + 2, f + 2), -1, dtype=np.int64)
        local[ii, jj] = np.arange(len(ii))
        tris = []
        for i in range(f):
            for j in range(f - i):
                tris.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                if i + j <= f - 2:
                    tris.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))
        return ii, jj, np.array(tris, dtype=np.int64).reshape(-1, 3)

    def _build(self, f: int) -> None:
        net = self.net
        self.frequency = f
        self.x_max, self.y_max = net.grid_size(f)
        ids = net.triangles.astype(np.int64)                     # (T, 3)
        corners = net.points[net.triangles]                     # (T, 3, 3)
        lat = net.lattice.astype(np.int64) * f                  # (T, 3, 2)
        o = lat[:, 0]
        e1 = (lat[:, 1] - o) // f
        e2 = (lat[:, 2] - o) // f
        ii, jj, tris = self._template(f)
        L, T = len(ii), len(ids)
        counts = np.stack([f - ii - jj, ii, jj], axis=1)        # (L, 3)

        gx = o[:, None, 0] + ii[None, :] * e1[:, None, 0] + jj[None, :] * e2[:, None, 0]   # (T, L)
        gy = o[:, None, 1] + ii[None, :] * e1[:, None, 1] + jj[None, :] * e2[:, None, 1]
        cell = (gx * (self.y_max + 1) + gy).ravel()

        # seam key of every point: its non-zero barycentric counts against the base corners
        n_base = len(net.points)
        pid = np.where(counts[None, :, :] > 0, ids[:, None, :], n_base)                 # (T, L, 3)
        cnt = np.where(counts[None, :, :] > 0, np.broadcast_to(counts, (T, L, 3)), 0)
        order = np.argsort(pid, axis=2, kind='stable')
        pid = np.take_along_axis(pid, order, axis=2)
        cnt = np.take_along_axis(cnt, order, axis=2)
        code = pid * (f + 1) + cnt
        base_ = (n_base + 1) * (f + 1)
        key = ((code[..., 0] * base_ + code[..., 1]) * base_ + code[..., 2]).ravel()

        # one stored position per grid cell, taken from the first triangle that reaches it
        cells, first, inverse = np.unique(cell, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        if (key != key[first][inverse]).any():
            bad = int(np.flatnonzero(key != key[first][inverse])[0])
            raise AssertionError(f'{self.base} net overlaps itself at grid '
                                 f'({int(gx.ravel()[bad])}, {int(gy.ravel()[bad])})')
        xy = np.stack([cells // (self.y_max + 1), cells % (self.y_max + 1)], axis=1).astype(np.int32)
        t_of, l_of = np.divmod(first, L)
        c = counts[l_of]
        p = corners[t_of]
        coords = (c[:, 0, None] * p[:, 0] + c[:, 1, None] * p[:, 1] + c[:, 2, None] * p[:, 2]) / f
        coords = _normalise_rows(coords)

        index = np.full((self.x_max + 1, self.y_max + 1), -1, dtype=np.int32)
        index[xy[:, 0], xy[:, 1]] = np.arange(len(xy), dtype=np.int32)

        # seam copies: stored positions with the same key, listed in creation order
        skey = key[first]
        uniq, cls, size = np.unique(skey, return_inverse=True, return_counts=True)
        cls = cls.reshape(-1)
        same: dict[int, list[int]] = {}
        seam = np.flatnonzero(size[cls] > 1)
        if len(seam):
            seam = seam[np.lexsort((first[seam], cls[seam]))]
            groups = np.split(seam, np.flatnonzero(np.diff(cls[seam])) + 1)
            for g in groups:
                g = g.tolist()
                for a in g:
                    same[a] = [b for b in g if b != a]

        # faces in the original order; which of the six offsets are edges of the net
        faces = inverse.reshape(T, L)[:, tris].reshape(-1, 3).astype(np.int32)
        a = faces.ravel()
        b = faces[:, [1, 2, 0]].ravel()
        dirs = np.full((3, 3), -1, dtype=np.int64)
        for k, (dx, dy) in enumerate(_DIRECTIONS):
            dirs[dx + 1, dy + 1] = k
        mask = np.zeros(len(xy), dtype=np.int64)
        for p_, q_ in ((a, b), (b, a)):
            k = dirs[xy[q_, 0] - xy[p_, 0] + 1, xy[q_, 1] - xy[p_, 1] + 1]
            np.bitwise_or.at(mask, p_, np.left_shift(1, k))
        cand = self._neighbour_candidates(xy, index)
        cand[(mask[:, None] >> np.arange(6)[None, :]) & 1 == 0] = -1
        self._mask = mask

        self.arcLength = _mean_edge_angle(coords, faces)   # mean great-circle edge length (radians)
        self._set_arrays(xy, index, coords, faces, cand, same, cls.astype(np.int32))

    def _after_materialise(self, flat) -> None:
        for v, m in zip(flat, self._mask.tolist(), strict=True):
            v.neighbour_mask = m


class TetrahedronDome(NetDome):
    """
    Geodesic dome on the tetrahedron, stored as the 4HSOM orthogonal array of
    R. M. de Sousa and R. C. L. Oliveira (IJCNN 2012): f + 1 rows (y = 0..f) by
    2f + 1 columns (x = 0..2f), every cell used.

        row f:   C . . . D . . . C       C at (0, f) and (2f, f); D (north pole) at (f, f)
        row 0:   A . . . B . . . A       A at (0, 0) and (2f, 0); B at (f, 0)

    Seams: column 0 = column 2f, and (i, 0) = (2f - i, 0), (i, f) = (2f - i, f).
    2f^2 + 2 unique points, 4f^2 triangles; the four corners A, B, C, D have 3 neighbours.
    """

    base = 'tetrahedron'


class DodecahedronDome(NetDome):
    """
    Geodesic dome on the dodecahedron: each pentagon is cut into 5 triangles at its centre
    (the pentakis dodecahedron) and each of those into an f-frequency grid, on the flat
    pentagon.  30f^2 + 2 unique points, 60f^2 triangles; the 12 pentagon centres have 5
    neighbours, every other point 6.  Frequency 1 is the pentakis dodecahedron (32 points).
    """

    base = 'dodecahedron'


_DOME_CLASSES = {'icosahedron': IcosahedronDome, 'tetrahedron': TetrahedronDome, 'dodecahedron': DodecahedronDome}
assert set(_DOME_CLASSES) == set(BASES)
