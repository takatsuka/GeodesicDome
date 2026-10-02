# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
01 -- Quick start: build geodesic domes and inspect the data structure.

Run:  python examples/01_quickstart.py
Needs: numpy
"""
import dome_utils  # noqa: F401  (adds the repo root to sys.path when run from a checkout)
import numpy as np
from dome_utils import unique_mesh

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

# ---------------------------------------------------------------------------
# 1. Build a dome.  frequency = number of segments each icosahedron edge is cut into.
# ---------------------------------------------------------------------------
dome = GeodesicDome(frequency=4)
print(dome)                                   # GeodesicDome(frequency=4)

# Equivalent, step by step (split() is cumulative: 2 x 2 = 4):
dome_b = GeodesicDome()                       # plain icosahedron, frequency 1
dome_b.split(2)
dome_b.split(2)
assert dome_b.frequency == dome.frequency == 4

# ---------------------------------------------------------------------------
# 2. Get the geometry as NumPy arrays.
# ---------------------------------------------------------------------------
xyz = dome.get_all_xyz()                              # (N, 3) points on the unit sphere
triangles = dome.get_all_triangles().reshape(-1, 3)   # (F, 3) indices into xyz
print(f'stored vertices : {xyz.shape[0]}')
print(f'triangles       : {triangles.shape[0]}')
print(f'all on unit sphere? {np.allclose(np.linalg.norm(xyz, axis=1), 1.0)}')

# Vertices on the seams of the unfolded net are stored more than once.
points, faces, index_map = unique_mesh(dome)
print(f'unique vertices : {points.shape[0]}   (10 f^2 + 2 = {10 * 4 ** 2 + 2})')

# ---------------------------------------------------------------------------
# 3. Individual vertices.
# ---------------------------------------------------------------------------
v = dome.get_vertex_at(6, 5)          # (x, y) = position on the rectilinear index grid
print('\nvertex at grid (6, 5)')
print(f'  id            : {v.id}')
print(f'  xyz           : {np.round(v.coord, 4)}')
print(f'  (colat, lon)  : {np.round(v.latlon_coord, 4)}  radians, measured from the +y axis')
print(f'  seam copies   : {len(v.same_vertices or [])}')

corner = dome.get_vertex_at(0, 0)     # one of the 12 original icosahedron corners
print('vertex at grid (0, 0) is an icosahedron corner; its seam copies are at',
      [(s.x, s.y) for s in corner.same_vertices])

# Attach any payload you like to a vertex (e.g. a weight vector for a SOM node).
v.set_data(np.random.default_rng(0).normal(size=3))
print(f'  data          : {np.round(v.data, 3)}')

# ---------------------------------------------------------------------------
# 4. How the dome grows with frequency.
# ---------------------------------------------------------------------------
print('\n freq | stored | unique |  faces | mean edge (unit sphere)')
print('------+--------+--------+--------+------------------------')
for f in (1, 2, 3, 4, 6, 8, 12):
    d = GeodesicDome(f)
    p, fc, _ = unique_mesh(d)
    edges = np.linalg.norm(p[fc[:, 0]] - p[fc[:, 1]], axis=1)
    print(f' {f:4d} | {len(d.get_all_vertices()):6d} | {len(p):6d} | {len(fc):6d} | {edges.mean():.4f}')
