# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
08 -- Flat grids: mt.geodesicdome.grid.plane.Plane

Plane implements the same Manifold interface as GeodesicDome, so code written against
get_neighbours / get_neighbours_in_distance / get_faces works with both.

    lattice : Lattice.Hexagonal (6 neighbours, odd rows shifted by 1/2)
              Lattice.Rectilinear (4 neighbours)
    topology: Topology.Plane (hard borders)
              Topology.Donut (edges wrap around, i.e. a torus; neighbour search only,
                              faces are not generated across the wrap)

Run:  python examples/08_plane_grids.py
Needs: numpy, matplotlib
Saves: examples/output/08_plane_grids.png
"""
import matplotlib.pyplot as plt
from dome_utils import output_path
from matplotlib.collections import PolyCollection

from mt.geodesicdome.grid.plane import Lattice, Plane, Topology

W, H = 12, 10          # use an even height for a hexagonal donut
RINGS = 3
colours = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60']

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for row, lattice in enumerate((Lattice.Hexagonal, Lattice.Rectilinear)):
    for col, topology in enumerate((Topology.Plane, Topology.Donut)):
        grid = Plane(W, H, lattice, topology)
        xy = grid.get_all_xyz()[:, :2]
        k = grid.get_number_of_vertices_per_face()
        faces = grid.get_all_triangles().reshape(-1, k)        # triangles or quads

        centre = grid.get_vertex_at(0, 4)                      # on the left border
        grid.unmark_vertices()
        rings = grid.get_neighbours_in_distance(centre, RINGS)

        ax = axes[row, col]
        ax.add_collection(PolyCollection(xy[faces], facecolors='#eef2f7', edgecolors='#9aa5b1', lw=0.6))
        ax.scatter(*xy.T, s=10, c='#9aa5b1')
        ax.scatter(*xy[centre.id], s=90, c=colours[0], zorder=3)
        for r, ring in enumerate(rings, start=1):
            ax.scatter(*xy[[u.id for u in ring]].T, s=50, c=colours[r], zorder=3)
        ax.autoscale(), ax.set_aspect('equal'), ax.set_axis_off()
        ax.set_title(f'{lattice.name} / {topology.name}\nring sizes {[len(r) for r in rings]}')

handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=name)
           for c, name in zip(colours, ['centre'] + [f'ring {r}' for r in range(1, RINGS + 1)], strict=False)]
fig.legend(handles=handles, loc='lower center', ncol=RINGS + 1)
fig.suptitle(f'Plane({W}, {H}): neighbour rings from a border vertex', fontsize=14)
fig.tight_layout(rect=(0, 0.05, 1, 1))
path = output_path('08_plane_grids.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
