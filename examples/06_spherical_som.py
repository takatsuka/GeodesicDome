# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
06 -- Showcase: a tiny Self-Organising Map (SOM) living on a geodesic sphere.

A spherical lattice has no borders, so every SOM node has the same neighbourhood
(apart from the 12 five-neighbour corners) -- no edge effects.  This example uses:

    * the dome vertices as SOM nodes,
    * `get_neighbours_in_distance` for the ring-shaped neighbourhood function,
    * `vertex.set_data` to store each node's weight vector,
    * a map projection to show the trained map.

The SOM learns random RGB colours, so the result is a smooth colour field on the sphere.

Run:  python examples/06_spherical_som.py
Needs: numpy, matplotlib
Saves: examples/output/06_spherical_som.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import neighbour_rings, output_path, unique_mesh
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.projection.equal_earth import EqualEarth

FREQ = 8                 # 642 SOM nodes
ITERATIONS = 3000
MAX_RADIUS = 6           # neighbourhood radius (in rings) at the start of training
rng = np.random.default_rng(42)

dome = GeodesicDome(FREQ)
points, faces, index_map = unique_mesh(dome)
vertices = dome.get_all_vertices()

# one representative stored vertex per physical node
representative = {}
for v in vertices:
    representative.setdefault(index_map[v.id], v)

weights = rng.random((len(points), 3))          # random initial RGB weight per node
initial = weights.copy()
samples = rng.random((ITERATIONS, 3))           # training data: random colours

for t, x in enumerate(samples):
    progress = t / ITERATIONS
    lr = 0.5 * (0.02 / 0.5) ** progress                     # 0.5 -> 0.02
    radius = max(1, round(MAX_RADIUS * (1 - progress)))     # 6 -> 1 rings
    sigma = max(radius / 2, 0.5)

    bmu = int(np.argmin(((weights - x) ** 2).sum(axis=1)))  # best matching unit
    weights[bmu] += lr * (x - weights[bmu])
    for r, ring in enumerate(neighbour_rings(dome, representative[bmu], radius), start=1):
        idx = np.unique([index_map[u.id] for u in ring])
        weights[idx] += lr * np.exp(-r * r / (2 * sigma * sigma)) * (x - weights[idx])

# store the trained weights on every stored vertex (seam copies share the value)
for v in vertices:
    v.set_data(weights[index_map[v.id]])

# mean distance between neighbouring nodes = how smooth the map is
edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
def roughness(w):
    return np.linalg.norm(w[edges[:, 0]] - w[edges[:, 1]], axis=1).mean()
print(f'mean neighbour colour distance: before {roughness(initial):.3f}  after {roughness(weights):.3f}')


# ------------------------------------------------------------------ plotting
def draw_map(ax, colour_of_vertex_id, title):
    triangles = EqualEarth().build(dome)
    flat = np.array([v.projected_coord for v in vertices])
    ax.add_collection(PolyCollection(flat[triangles], facecolors=colour_of_vertex_id[triangles].mean(axis=1),
                                     edgecolors='none'))
    ax.autoscale(), ax.set_aspect('equal'), ax.set_axis_off(), ax.set_title(title)


def draw_globe(ax, node_colours, azim, title):
    tri = points[faces]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=node_colours[faces].mean(axis=1), edgecolors='none'))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)), ax.view_init(elev=15, azim=azim), ax.set_axis_off(), ax.set_title(title)


per_vertex_initial = initial[index_map]                     # colour for every stored vertex id
per_vertex_trained = np.array([v.data for v in vertices])   # read back what we stored

fig = plt.figure(figsize=(16, 4.6))
draw_map(fig.add_subplot(1, 4, 1), per_vertex_initial, 'before training (Equal Earth)')
draw_map(fig.add_subplot(1, 4, 2), per_vertex_trained, f'after {ITERATIONS} samples (Equal Earth)')
draw_globe(fig.add_subplot(1, 4, 3, projection='3d'), weights, 30, 'trained, front')
draw_globe(fig.add_subplot(1, 4, 4, projection='3d'), weights, 210, 'trained, back')
fig.suptitle(f'Spherical SOM with {len(points)} nodes on GeodesicDome(frequency={FREQ})', fontsize=14)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.82, wspace=0.05)
path = output_path('06_spherical_som.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
