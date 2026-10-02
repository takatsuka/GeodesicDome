# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
07 -- Export a dome as a mesh file (OBJ, OFF, NumPy .npz).

The exported mesh is de-duplicated (every physical point once) and every triangle is
wound counter-clockwise seen from outside, so normals point outwards.
Open the .obj / .off in Blender, MeshLab, three.js, trimesh, etc.

Run:  python examples/07_export_mesh.py [frequency] [radius]
Needs: numpy
Saves: examples/output/geodesic_f<freq>.obj / .off / .npz
"""
import sys

import numpy as np
from dome_utils import output_path, unique_mesh

from mt.geodesicdome.grid.geodesicdome import GeodesicDome

freq = int(sys.argv[1]) if len(sys.argv) > 1 else 6
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

dome = GeodesicDome(freq)
points, faces, _ = unique_mesh(dome)
points = points * radius

# sanity check: Euler characteristic of a sphere is V - E + F = 2
edges = {tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
print(f'V={len(points)}  E={len(edges)}  F={len(faces)}  V-E+F={len(points) - len(edges) + len(faces)}')

base = output_path(f'geodesic_f{freq}')

with open(base + '.obj', 'w') as fh:                     # OBJ: 1-based indices
    fh.write(f'# geodesic dome, frequency {freq}, radius {radius}\n')
    for x, y, z in points:
        fh.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
    for a, b, c in faces + 1:
        fh.write(f'f {a} {b} {c}\n')

with open(base + '.off', 'w') as fh:                     # OFF: 0-based indices
    fh.write(f'OFF\n{len(points)} {len(faces)} {len(edges)}\n')
    for x, y, z in points:
        fh.write(f'{x:.9f} {y:.9f} {z:.9f}\n')
    for a, b, c in faces:
        fh.write(f'3 {a} {b} {c}\n')

np.savez(base + '.npz', points=points, faces=faces)      # reload with np.load(...)

for ext in ('.obj', '.off', '.npz'):
    print(f'saved {base}{ext}')
