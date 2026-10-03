# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
10 -- Geodesic domes on three base polyhedra: tetrahedron, icosahedron, dodecahedron.

    GeodesicDome(f, base='tetrahedron')    2 f^2 + 2 points  (4HSOM orthogonal array, de Sousa & Oliveira 2012)
    GeodesicDome(f, base='icosahedron')   10 f^2 + 2 points  (the default; Wu & Takatsuka 2006)
    GeodesicDome(f, base='dodecahedron')  30 f^2 + 2 points  (pentakis dodecahedron)

Top row: the domes on the sphere.  Bottom row: the nets they are indexed on.  All three use
the same index grid, where the six neighbours of (x, y) are the offsets
(+1, 0) (-1, 0) (0, +1) (0, -1) (+1, +1) (-1, -1); here the grid is sheared so that every
triangle is drawn equilateral.  Orange points are stored more than once (seam copies), red
ones are the corners of the base solid (3 or 5 neighbours instead of 6).

Run:  python examples/10_base_polyhedra.py
Needs: numpy, matplotlib
Saves: examples/output/10_base_polyhedra.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import output_path, shade, unique_mesh
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

FREQ = 4
BASES = ('tetrahedron', 'icosahedron', 'dodecahedron')
COLOUR = {'tetrahedron': (0.80, 0.42, 0.20), 'icosahedron': (0.27, 0.51, 0.71), 'dodecahedron': (0.30, 0.62, 0.38)}
shear = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])   # grid (x, y) -> plane

fig = plt.figure(figsize=(16, 9.5))
for n, base in enumerate(BASES):
    dome = GeodesicDome(FREQ, base=base)
    vertices = dome.get_all_vertices()
    stored = dome.get_all_triangles().reshape(-1, 3)        # also assigns vertex.id
    points, faces, index_map = unique_mesh(dome)

    # 3-D view
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    ax = fig.add_subplot(2, 3, n + 1, projection='3d')
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(normals)[:, None] * np.array(COLOUR[base]),
                                         edgecolors='white', linewidths=0.4))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()
    ax.set_title(f"GeodesicDome({FREQ}, base='{base}')\n{len(points)} points, {len(faces)} triangles")

    # the net on the index grid
    degree = np.zeros(len(points), dtype=int)
    for a, b in {tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}:
        degree[a] += 1
        degree[b] += 1
    corner = degree[index_map] < 6
    seam = np.array([bool(v.same_vertices) for v in vertices]) & ~corner
    net = np.array([[v.x, v.y] for v in vertices], dtype=float) @ shear
    ax = fig.add_subplot(2, 3, n + 4)
    ax.add_collection(PolyCollection(net[stored], facecolors=(*COLOUR[base], 0.18),
                                     edgecolors=COLOUR[base], linewidths=0.5))
    ax.scatter(*net[~seam & ~corner].T, s=6, c='#4a4a4a', zorder=3)
    ax.scatter(*net[seam].T, s=16, c='#e6892e', zorder=3, label='seam copy')
    ax.scatter(*net[corner].T, s=40, c='#c0392b', marker='p', zorder=4,
               label=f'corner ({degree[index_map][corner][0]} neighbours)')
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_axis_off()
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_title(f'net: grid {dome.x_max + 1} x {dome.y_max + 1}, {len(vertices)} stored vertices')

fig.suptitle('GeodesicDome: the same index grid on three base polyhedra', fontsize=14)
fig.tight_layout()
path = output_path('10_base_polyhedra.png')
fig.savefig(path, dpi=110)
print(f'saved {path}')
plt.show()
