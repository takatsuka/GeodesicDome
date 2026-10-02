# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
04 -- Neighbourhood queries on the sphere.

`get_neighbours(v)` returns the immediate neighbours of a vertex and
`get_neighbours_in_distance(v, d)` returns d "rings" around it.  Seam copies are
handled for you, so a ring continues seamlessly across the edges of the net.

    interior / seam vertex : ring sizes 6, 12, 18, 24, ...
    icosahedron corner     : ring sizes 5, 10, 15, 20, ...

IMPORTANT: the search marks vertices as visited.  Call `dome.unmark_vertices()`
before every new query (the helper `neighbour_rings` does this for you).

Run:  python examples/04_neighbours.py
Needs: numpy, matplotlib
Saves: examples/output/04_neighbours.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import neighbour_rings, output_path, unique_mesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

FREQ = 10
RINGS = 4
dome = GeodesicDome(FREQ)
points, faces, index_map = unique_mesh(dome)       # also assigns vertex.id

# --- the basic API ---------------------------------------------------------
v = dome.get_vertex_at(15, 17)
dome.unmark_vertices()
first = dome.get_neighbours(v, False)
print(f'vertex ({v.x},{v.y}) has {len(first)} neighbours at grid positions',
      [(n.x, n.y) for n in first])

# --- pick three instructive vertices --------------------------------------
corner = dome.get_vertex_at(0, 0)                               # icosahedron corner
seam = dome.get_vertex_at(0, FREQ // 2)                         # middle of a net edge
cases = [('interior vertex', v), ('seam vertex (stored twice)', seam), ('icosahedron corner', corner)]

colours = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60', '#2980b9']
fig = plt.figure(figsize=(15, 5.2))
for n, (label, centre) in enumerate(cases, start=1):
    rings = neighbour_rings(dome, centre, RINGS)
    sizes = [len(r) for r in rings]
    print(f'{label:28s} grid ({centre.x:2d},{centre.y:2d})  ring sizes {sizes}')

    # ring number of every unique point (RINGS + 1 = outside the neighbourhood)
    rank = np.full(len(points), RINGS + 1)
    rank[index_map[centre.id]] = 0
    for r, ring in enumerate(rings, start=1):
        rank[[index_map[u.id] for u in ring]] = r

    palette = np.array([to_rgb(c) for c in colours] + [(0.85, 0.85, 0.85)])
    point_colour = palette[rank]
    face_rank = rank[faces].max(axis=1)                  # a face belongs to its outermost ring
    face_colour = 0.5 + 0.5 * palette[face_rank]         # lighter tint of the ring colour
    face_colour[face_rank > RINGS] = (0.93, 0.94, 0.96)

    # draw only the hemisphere facing the camera
    view = centre.coord / np.linalg.norm(centre.coord)
    tri = points[faces]
    visible = tri.mean(axis=1) @ view > 0.05

    ax = fig.add_subplot(1, 3, n, projection='3d')
    ax.add_collection3d(Poly3DCollection(tri[visible], facecolors=face_colour[visible],
                                         edgecolors='#9aa5b1', linewidths=0.3))
    front = points @ view > 0.05
    ax.scatter(*points[front].T, c=point_colour[front], s=12, depthshade=False)
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=np.degrees(np.arcsin(view[2])), azim=np.degrees(np.arctan2(view[1], view[0])))
    ax.set_axis_off()
    ax.set_title(f'{label}\nring sizes {sizes}')

handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=name)
           for c, name in zip(colours, ['centre'] + [f'ring {r}' for r in range(1, RINGS + 1)], strict=False)]
fig.legend(handles=handles, loc='lower center', ncol=RINGS + 1)
fig.suptitle(f'get_neighbours_in_distance(v, {RINGS}) on GeodesicDome(frequency={FREQ})', fontsize=14)
fig.subplots_adjust(left=0, right=1, bottom=0.1, top=0.83, wspace=0)
path = output_path('04_neighbours.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
