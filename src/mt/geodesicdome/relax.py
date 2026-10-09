# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Lloyd relaxation of a spherical triangle mesh with its connectivity held fixed.

Every point moves to the centroid of its dual cell (the polygon of the circumcentres of the
triangles around it), repeatedly.  Applied to a geodesic dome it keeps every grid position,
neighbour and seam copy, and only evens out the cell areas: the plain icosahedral dome's cells
vary in area by a factor of about 2.2 (they are stretched near the face centres), the relaxed
dome's by much less.  Use it through :meth:`GeodesicDome.relax` or
``GeodesicDome(frequency, relax=True)``; :func:`lloyd` works on any (points, triangles) mesh.

Holding the triangles fixed (rather than recomputing Voronoi cells) and over-relaxing are this
implementation's choices; they are not part of Lloyd's method.

* Fixed connectivity.  The dual cell of a triangulation is the Voronoi cell as long as the
  triangulation stays Delaunay.  For the icosahedral dome the Delaunay triangulation of the
  relaxed points was checked to be the dome's own at N = 162 ... 40 962, so the result is a
  spherical centroidal Voronoi tessellation that keeps the dome's graph.  The same holds for the
  tetrahedral and dodecahedral domes at the sizes in ``tests/test_relax.py``.  The cell construction
  assumes each circumcentre lies inside or near its triangle, which holds for nearly equilateral
  triangles.
* Over-relaxation.  Each step moves a point ``omega`` times the way to its cell centroid.  A
  converged point is its own centroid whatever ``omega`` is, so this changes the speed, not the
  answer.  ``omega = 1.8`` (chosen by trial) reaches convergence at N = 40 962 in minutes, where
  plain Lloyd (``omega = 1``) converges in 100 steps only up to N of about 2 562.  ``omega`` close
  to 2 can overshoot; convergence is checked directly through ``tol``.

The method is the one used in the ``spiral`` repository's lattice comparison
(``benchmarks/lattice_comparison/relax.py``), with every step done in one batch of array operations
so that it runs on the compute backend (:mod:`mt.geodesicdome.backend`): a CUDA or Apple (MPS) GPU
when one is available, otherwise NumPy on the CPU.

* GPU precision.  On CUDA the relaxation runs in float64 (``backend=None`` asks for it).  The
  Apple GPU (MPS) has only float32, whose rounding stops the largest move from shrinking below a
  few 1e-6 rad.  A float32 run therefore continues until its largest move has not reached a new
  low for ``PATIENCE`` steps, and the last steps (typically a few dozen) are made on the CPU in
  float64, so the result meets ``tol`` as a CPU-only run does.
* Small meshes run on the CPU even when a GPU is present (``backend=None``): below
  ``GPU_MIN_POINTS`` points the per-step launch overhead outweighs the gain.

References:
    S. P. Lloyd, "Least squares quantization in PCM", IEEE Trans. Inf. Theory 28(2):129-137, 1982.
        doi:10.1109/TIT.1982.1056489
    Q. Du, V. Faber and M. Gunzburger, "Centroidal Voronoi tessellations: applications and algorithms",
        SIAM Review 41(4):637-676, 1999.  doi:10.1137/S0036144599352836
    Q. Du, M. D. Gunzburger and L. Ju, "Constrained centroidal Voronoi tessellations for surfaces",
        SIAM J. Sci. Comput. 24(5):1488-1506, 2003.  doi:10.1137/S1064827501391576
"""
from __future__ import annotations

import numpy as np

__all__ = ['lloyd', 'triangle_areas', 'cell_areas', 'outward', 'CONVERGED', 'GPU_MIN_POINTS', 'PATIENCE']

#: settings that run the relaxation to convergence: no point moves 1e-7 rad in a step
CONVERGED = {'iters': 20000, 'omega': 1.8, 'tol': 1e-7}
#: with ``backend=None``, meshes with fewer points than this stay on the CPU
GPU_MIN_POINTS = 5000
#: a float32 run hands over to float64 on the CPU once its largest move has not fallen for this many steps
PATIENCE = 50


def _unit(p: np.ndarray) -> np.ndarray:
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def triangle_areas(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Spherical areas (steradians) of triangles (F, 3) on unit vectors x (Van Oosterom-Strackee)."""
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    num = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    den = 1.0 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    return 2.0 * np.arctan2(num, den)


def cell_areas(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(N,) dual (barycentric) cell area of every point: a third of the area of each triangle around it."""
    x = np.asarray(x, float)
    faces = np.asarray(faces, np.int64)
    out = np.zeros(len(x))
    np.add.at(out, faces.ravel(), np.repeat(triangle_areas(x, faces) / 3.0, 3))
    return out


def outward(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """`faces` with every triangle wound counter-clockwise seen from outside the sphere."""
    faces = np.array(faces, dtype=np.int64, copy=True)
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    flip = np.einsum('ij,ij->i', np.cross(b - a, c - a), a + b + c) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


# ---------------------------------------------------------------------------------- one step, any backend
class _Ops:
    """The few array operations a Lloyd step needs, on numpy / cupy (``xp``) or torch arrays."""

    def __init__(self, b):
        self.b = b
        if b.name == 'torch':
            t = b.torch
            self.cross = lambda u, v: t.linalg.cross(u, v, dim=1)
            self.atan2 = t.atan2
            self.arcsin = t.asin
            self.cat = lambda a: t.cat(a, dim=0)
            self.amax = lambda a: float(a.max())
        else:
            xp = b.xp
            self.cross = lambda u, v: xp.cross(u, v)
            self.atan2 = xp.arctan2
            self.arcsin = xp.arcsin
            self.cat = xp.concatenate
            self.amax = lambda a: float(a.max())

    @staticmethod
    def dot(u, v):
        return (u * v).sum(1)

    def unit(self, p):
        return p / self.b.sqrt(self.dot(p, p))[:, None]


def _run(b, x, tri, iters, omega, tol, patience=0):
    """
    Lloyd steps on backend `b` (x already converted); returns (x on the backend, steps, last move).
    With `patience` > 0 it also stops once the largest move has not reached a new low for that many
    steps (the precision floor of a float32 device).
    """
    ops = _Ops(b)
    n = x.shape[0]
    # the six half-edge corners of every triangle: corner k, towards corner m
    pairs = [(k, m) for k in range(3) for m in ((k + 1) % 3, (k + 2) % 3)]
    v = b.as_index(np.concatenate([tri[:, k] for k, _ in pairs]))
    w = b.as_index(np.concatenate([tri[:, m] for _, m in pairs]))
    t0, t1, t2 = (b.as_index(tri[:, k]) for k in range(3))
    reps = len(pairs)
    steps, moved, best, since = 0, float('inf'), float('inf'), 0
    for _ in range(iters):
        steps += 1
        a, bb, c = x[t0], x[t1], x[t2]
        cc = ops.unit(ops.cross(bb - a, c - a))            # circumcentre of each spherical triangle
        cc = ops.cat([cc] * reps)
        p = x[v]
        mid = ops.unit(p + x[w])
        # area of the small spherical triangle (p, cc, mid), Van Oosterom-Strackee
        num = b.abs(ops.dot(p, ops.cross(cc, mid)))
        den = 1.0 + ops.dot(p, cc) + ops.dot(cc, mid) + ops.dot(mid, p)
        area = 2.0 * ops.atan2(num, den)
        tot = b.index_add(b.zeros((n, 3)), v, area[:, None] * ops.unit(p + cc + mid))
        wsum = b.index_add(b.zeros((n,)), v, area)
        target = ops.unit(tot / wsum[:, None])
        new = ops.unit(x + omega * (target - x))
        chord = b.sqrt(ops.dot(new - x, new - x))
        moved = 2.0 * ops.amax(ops.arcsin(b.clip(chord * 0.5, 0.0, 1.0)))   # largest move, radians
        x = new
        if moved < tol:
            break
        if patience:
            if moved < 0.98 * best:
                best, since = moved, 0
            else:
                since += 1
                if since >= patience:
                    break
    return x, steps, moved


def lloyd(x: np.ndarray, tri: np.ndarray, iters: int = 100, omega: float = 1.0, tol: float = 0.0,
          return_steps: bool = False, backend=None):
    """
    (N, 3) points after Lloyd steps on the fixed triangles `tri` (outward winding, see :func:`outward`).

    :param x: (N, 3) unit vectors, one per distinct point (no seam copies)
    :param tri: (F, 3) triangles as indices into `x`
    :param iters: maximum number of steps
    :param omega: over-relaxation: each point moves omega times the way to its cell centroid
                  (1 = plain Lloyd)
    :param tol: stop early once no point moves more than `tol` radians in a step
    :param return_steps: also return the number of steps taken
    :param backend: where to compute (see :func:`mt.geodesicdome.backend.get_backend`): None picks
                    the GPU when one is available and the mesh has at least ``GPU_MIN_POINTS``
                    points, else NumPy; 'cpu' / 'numpy', 'cuda', 'mps', 'torch', 'cupy' or a Backend
                    choose explicitly
    :return: the relaxed points (float64 numpy array), or (points, steps) if `return_steps`
    """
    from mt.geodesicdome.backend import get_backend

    x = np.asarray(x, np.float64)
    tri = np.asarray(tri, np.int64)
    if backend is None:
        b = get_backend()
        if b.is_gpu and len(x) < GPU_MIN_POINTS:
            b = get_backend('numpy')
        elif b.is_gpu and b.device.startswith('cuda') and b.dtype != np.float64:
            b = get_backend(b.spec, dtype='float64')          # CUDA has float64: no CPU finish needed
    else:
        b = get_backend(backend)
    steps = 0
    if iters > 0 and b.dtype != np.float64:
        # low-precision device: run there until its precision floor, then finish on the CPU in float64
        y, steps, moved = _run(b, b.asarray(x), tri, iters, omega, tol, patience=PATIENCE)
        x = _unit(b.to_numpy(y).astype(np.float64))
        if moved < tol:
            iters = steps
        b = get_backend('numpy')
    if iters - steps > 0:
        y, more, _ = _run(b, b.asarray(x), tri, iters - steps, omega, tol)
        x = np.asarray(b.to_numpy(y), np.float64)
        steps += more
    return (x, steps) if return_steps else x
