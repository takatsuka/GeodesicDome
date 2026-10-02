# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
03 -- Visualise the unfolded net that GeodesicDome uses for indexing.

Every vertex has an integer grid position (vertex.x, vertex.y).  Neighbours on the
sphere are always one of the six grid offsets
    (+1, 0) (-1, 0) (0, +1) (0, -1) (+1, +1) (-1, -1)
which is what makes neighbour search fast.  Vertices on the boundary of the net are
stored several times (``vertex.same_vertices``) so the net can be "glued" back into a
sphere.

Run:  python examples/03_unfolded_net.py
Needs: numpy, matplotlib
Saves: examples/output/03_unfolded_net.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import output_path, unique_mesh
from matplotlib.collections import PolyCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

FREQ = 3
dome = GeodesicDome(FREQ)
vertices = dome.get_all_vertices()
triangles = dome.get_all_triangles().reshape(-1, 3)      # also assigns vertex.id

grid = np.array([[v.x, v.y] for v in vertices], dtype=float)

# Shear the index grid so that every triangle becomes equilateral:
#   (x, y) -> x * (1, 0) + y * (-1/2, sqrt(3)/2)
shear = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])
net = grid @ shear

# Classify vertices: the 12 icosahedron corners have 5 neighbours on the sphere.
points, faces, index_map = unique_mesh(dome)
degree = np.zeros(len(points), dtype=int)
edges = {tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
for a, b in edges:
    degree[a] += 1
    degree[b] += 1
is_corner = degree[index_map] == 5
is_seam = np.array([bool(v.same_vertices) for v in vertices]) & ~is_corner
is_inner = ~is_corner & ~is_seam

fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.25]})
for ax, coords, title in ((axes[0], grid, 'index grid  (vertex.x, vertex.y)'),
                          (axes[1], net, 'same grid, sheared: the icosahedron net')):
    ax.add_collection(PolyCollection(coords[triangles], facecolors='#e8eef6',
                                     edgecolors='#8aa4c8', linewidths=0.6))
    ax.scatter(*coords[is_inner].T, s=14, c='#4a6fa5', label='interior vertex', zorder=3)
    ax.scatter(*coords[is_seam].T, s=26, c='#e6892e', label='seam vertex (stored 2x)', zorder=3)
    ax.scatter(*coords[is_corner].T, s=60, c='#c0392b', marker='p',
               label='icosahedron corner (5 neighbours)', zorder=4)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.autoscale()

# annotate a vertex and its six grid neighbours on the index grid
v = dome.get_vertex_at(4, 4)
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
    axes[0].annotate('', xy=(v.x + 0.8 * dx, v.y + 0.8 * dy), xytext=(v.x, v.y),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
axes[0].text(v.x + 0.15, v.y - 0.45, f'({v.x},{v.y})', fontsize=9)
axes[0].legend(loc='upper left', fontsize=8)

fig.suptitle(f'GeodesicDome(frequency={FREQ}): {len(vertices)} stored vertices, '
             f'{len(points)} unique points, {len(triangles)} triangles')
fig.tight_layout()
path = output_path('03_unfolded_net.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
