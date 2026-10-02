# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Array kernels on geodesic domes that run on the GPU when available (see mt.geodesicdome.backend).

    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    from mt.geodesicdome import compute

    mesh = compute.DomeArrays.from_dome(GeodesicDome(32))      # 10 242 unique vertices
    d = compute.angular_distance(mesh.points)                  # (V, V) great-circle angles, on the GPU
    idx = compute.nearest_vertex(mesh.points, queries)         # nearest dome vertex of every query point

Large results are computed in blocks that fit in device memory, and on the CPU the blocks run
on every core.  Results come back as numpy arrays unless ``as_numpy=False`` is given, in which
case they stay on the device as backend arrays (torch tensors, cupy arrays or numpy arrays).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mt.geodesicdome.backend import Backend, get_backend

__all__ = ['DomeArrays', 'angular_distance', 'ring_distance', 'nearest_vertex', 'edges_from_faces']


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """(E, 2) sorted, unique vertex pairs along the edges of (F, k) faces."""
    faces = np.asarray(faces, dtype=np.int64)
    k = faces.shape[1]
    e = np.sort(np.vstack([faces[:, [i, (i + 1) % k]] for i in range(k)]), axis=1)
    return np.unique(e[e[:, 0] != e[:, 1]], axis=0)


@dataclass
class DomeArrays:
    """
    A geodesic dome as plain arrays (seam copies merged).

        points      (V, 3) unit vectors
        faces       (F, 3) vertex indices
        edges       (E, 2) neighbouring vertex pairs
        index_map   (N,)   row in `points` of every stored vertex (row = vertex.id)
        ring_length mean angle (radians) between neighbouring vertices
    """
    points: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    index_map: np.ndarray
    ring_length: float

    @classmethod
    def from_dome(cls, dome) -> DomeArrays:
        from mt.geodesicdome.interactive.mesh import unique_mesh

        points, faces, index_map = unique_mesh(dome)
        edges = edges_from_faces(faces)
        cos = np.einsum('ij,ij->i', points[edges[:, 0]], points[edges[:, 1]])
        ring = float(np.arccos(np.clip(cos, -1.0, 1.0)).mean())
        return cls(points, faces, edges, index_map, ring)

    @property
    def n(self) -> int:
        return len(self.points)


def _angles_block(b: Backend, a_dev, b_dev_t):
    return b.arccos(b.clip(a_dev @ b_dev_t, -1.0, 1.0))


def angular_distance(a: np.ndarray, other: np.ndarray | None = None, backend: Backend | str | None = None,
                     as_numpy: bool = True, scale: float = 1.0):
    """
    Great-circle angle (radians) between every unit vector in `a` (n, 3) and every one in
    `other` (m, 3; default `a`), divided by `scale`: an (n, m) matrix.
    """
    b = get_backend(backend)
    ad = b.asarray(a)
    od_t = (ad if other is None else b.asarray(other)).T
    m = od_t.shape[1]
    inv = 1.0 / float(scale)

    def block(s):
        d = _angles_block(b, ad[s], od_t)
        return d * inv if inv != 1.0 else d

    rows = b.block_rows(m, arrays=2, total=len(ad))
    parts = b.map_blocks(block, len(ad), rows)
    if as_numpy:
        return np.concatenate([b.to_numpy(p) for p in parts], axis=0) if parts else np.zeros((0, m))
    return b.cat(parts, axis=0) if parts else b.zeros((0, m))


def ring_distance(dome_arrays: DomeArrays, backend: Backend | str | None = None, as_numpy: bool = True):
    """(V, V) great-circle distances between dome vertices in rings (1 = neighbour spacing)."""
    return angular_distance(dome_arrays.points, backend=backend, as_numpy=as_numpy, scale=dome_arrays.ring_length)


def nearest_vertex(points: np.ndarray, queries: np.ndarray, k: int = 1, backend: Backend | str | None = None,
                   return_angle: bool = False):
    """
    Index of the nearest of `points` (V, 3) to each query direction (Q, 3); queries need not
    be normalised.  With k > 1 returns (Q, k) indices, nearest first.  With return_angle=True
    also returns the great-circle angles (radians).
    """
    b = get_backend(backend)
    q = np.asarray(queries, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-300)
    pts_t = b.asarray(points).T
    qd = b.asarray(q.reshape(-1, 3))

    def block(s):
        cos = qd[s] @ pts_t
        idx = b.argmax(cos, axis=1)[:, None] if k == 1 else b.largest(cos, k)
        if not return_angle:
            return b.to_numpy(idx), None
        best = b.to_numpy(b.gather(cos, idx))
        return b.to_numpy(idx), np.arccos(np.clip(best, -1.0, 1.0))

    rows = b.block_rows(pts_t.shape[1], arrays=2, total=len(qd))
    parts = b.map_blocks(block, len(qd), rows)
    idx = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, k), dtype=np.int64)
    shape = q.shape[:-1] + ((k,) if k > 1 else ())
    idx = idx.reshape(shape).astype(np.int64)
    if not return_angle:
        return idx
    ang = np.concatenate([p[1] for p in parts]).reshape(shape) if parts else np.zeros(shape)
    return idx, ang
