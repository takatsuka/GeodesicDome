# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
09 -- Interactive map projection: drag on the map to rotate the sphere.

    python examples/09_interactive_projection.py
    python examples/09_interactive_projection.py --freq 12 --projection "Wagner VI" --colors icosahedron
    python examples/09_interactive_projection.py --colors bumps --view -34 151
    python examples/09_interactive_projection.py --gif examples/output/09_rotation.gif   # no window

Mouse / keys (see mt.geodesicdome.interactive.viewer):
    drag            rotate the sphere (left-right spins, up-down tilts)
    arrows          rotate by 5 degrees (shift: 1 degree);  , / .  roll
    r  reset    p  next projection    g  graticule on/off    e  edges on/off

Needs: numpy, matplotlib with a GUI backend (a normal desktop Python is fine;
in Jupyter use `%matplotlib widget`).  --gif also needs pillow.
"""
import argparse

import dome_utils  # noqa: F401  (adds the repo root to sys.path when run from a checkout)
import numpy as np

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.interactive import ProjectionViewer
from mt.geodesicdome.interactive.viewer import DEFAULT_PROJECTIONS

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--freq', type=int, default=8, help='dome frequency (default 8)')
parser.add_argument('--projection', default='Equal Earth', choices=list(DEFAULT_PROJECTIONS))
parser.add_argument('--colors', default='position', choices=['position', 'icosahedron', 'bumps', 'data'],
                    help="face colouring; 'bumps' = a scalar field, 'data' = vertex.set_data demo")
parser.add_argument('--view', type=float, nargs=2, metavar=('LAT', 'LON'), help='initial centre of the map')
parser.add_argument('--no-edges', action='store_true')
parser.add_argument('--gif', metavar='PATH', help='record a rotation to a GIF instead of opening a window')
args = parser.parse_args()

dome = GeodesicDome(args.freq)
colors = args.colors
cmap = 'viridis'
if colors == 'bumps':
    # a smooth scalar field on the sphere: a few Gaussian bumps around random centres
    xyz = dome.get_all_xyz()
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(6, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    colors = np.exp(-(1 - xyz @ centres.T) * 6).sum(axis=1)     # one value per stored vertex
    cmap = 'magma'
elif colors == 'data':
    # anything stored with vertex.set_data can be shown -- here an RGB colour per vertex
    for v in dome.get_all_vertices():
        v.set_data(0.5 + 0.5 * np.array([np.sin(3 * v.coord[0]), v.coord[1], np.cos(2 * v.coord[2])]))

viewer = ProjectionViewer(dome, args.projection, colors=colors, cmap=cmap,
                          edges=False if args.no_edges else None, view=args.view,
                          title=f'GeodesicDome(frequency={args.freq}) — drag the map to rotate the sphere')

# react to rotations, e.g. to update another plot (here: just report where we are)
viewer.on_rotate(lambda vw: None)

if args.gif:
    import matplotlib.animation as animation

    frames = 72

    def step(i):
        # spin 360 degrees while slowly tilting the pole towards the viewer and back
        viewer.rotate(d_lon=360 / frames, d_lat=35 * np.cos(2 * np.pi * i / frames) * 2 * np.pi / frames)
        return []

    anim = animation.FuncAnimation(viewer.fig, step, frames=frames, interval=60, blit=False)
    anim.save(args.gif, writer=animation.PillowWriter(fps=15), dpi=55)
    print(f'saved {args.gif}')
else:
    viewer.show()
