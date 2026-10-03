# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
02 -- Render geodesic domes of increasing frequency in 3D.

Run:  python examples/02_plot_dome_3d.py
Needs: numpy, matplotlib
Saves: examples/output/02_domes_3d.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import output_path, shade, unique_mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

FREQUENCIES = (1, 2, 4, 8)

fig = plt.figure(figsize=(14, 4.4))
for n, f in enumerate(FREQUENCIES, start=1):
    dome = GeodesicDome(f)
    points, faces, _ = unique_mesh(dome)

    tri = points[faces]                                   # (F, 3, 3) triangle corners
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # colour each face with simple Lambert shading
    base = np.array([0.27, 0.51, 0.71])
    colours = shade(normals)[:, None] * base

    ax = fig.add_subplot(1, len(FREQUENCIES), n, projection='3d')
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colours, edgecolors='white', linewidths=0.4))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()
    ax.set_title(f'frequency {f}\n{len(points)} vertices, {len(faces)} faces')

fig.suptitle('GeodesicDome: GeodesicDome (icosahedron-based)', fontsize=14)
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.80, wspace=0)
path = output_path('02_domes_3d.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
