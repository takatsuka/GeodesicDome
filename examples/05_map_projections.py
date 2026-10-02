# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
05 -- Flatten the dome with the built-in map projections.

Available projections (all in mt.geodesicdome.projection):
    KavrayskiyVII   kavrayskiy.py
    WagnerVI        wagner.py
    WagnerIII       wagner.py
    EqualEarth      equal_earth.py

`projection.build(dome)` projects every vertex (stored in `vertex.projected_coord`)
and returns an (M, 3) array of triangle indices.  Triangles that would wrap around
the +-180 degree meridian are dropped, so M is a little smaller than 20 f^2.
The projections treat +z as the north pole (latitude = arcsin(z)).

Run:  python examples/05_map_projections.py
Needs: numpy, matplotlib
Saves: examples/output/05_map_projections.png
"""
import matplotlib.pyplot as plt
import numpy as np
from dome_utils import output_path
from matplotlib.collections import PolyCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.projection.equal_earth import EqualEarth
from mt.geodesicdome.projection.kavrayskiy import KavrayskiyVII
from mt.geodesicdome.projection.wagner import WagnerIII, WagnerVI

FREQ = 8
dome = GeodesicDome(FREQ)
xyz = dome.get_all_xyz()

# Colour every small triangle by the icosahedron face (1 of 20) it came from.
ico = GeodesicDome(1)
ico_centres = ico.get_all_xyz()[ico.get_all_triangles().reshape(-1, 3)].mean(axis=1)

fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
for ax, projection in zip(axes.flat, (KavrayskiyVII(), WagnerVI(), WagnerIII(), EqualEarth()), strict=False):
    triangles = projection.build(dome)                                    # (M, 3) vertex ids
    flat = np.array([v.projected_coord for v in dome.get_all_vertices()])  # (N, 2), row = vertex.id

    centres = xyz[triangles].mean(axis=1)
    parent_face = np.argmax(centres @ ico_centres.T, axis=1)
    colours = plt.cm.tab20(parent_face % 20)

    ax.add_collection(PolyCollection(flat[triangles], facecolors=colours,
                                     edgecolors='white', linewidths=0.3))
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'{type(projection).__name__}  ({len(triangles)} of {20 * FREQ ** 2} triangles)')

fig.suptitle(f'GeodesicDome(frequency={FREQ}) in four map projections, '
             'coloured by parent icosahedron face', fontsize=14)
fig.tight_layout()
path = output_path('05_map_projections.png')
fig.savefig(path, dpi=130)
print(f'saved {path}')
plt.show()
