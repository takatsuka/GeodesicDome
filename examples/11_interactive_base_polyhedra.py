# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
11 -- Interactive explorer for every type of geodesic dome: tetrahedron, icosahedron, dodecahedron.

    python examples/11_interactive_base_polyhedra.py
    python examples/11_interactive_base_polyhedra.py --base tetra --freq 6 --rings 4
    python examples/11_interactive_base_polyhedra.py --colouring "cell area"
    python examples/11_interactive_base_polyhedra.py --save examples/output/11_explorer.png   # no window

One window, three linked views of the same dome:

    3-D sphere          drag to turn it
    index-grid net      click a point to make it the centre
    map projection      centred on that vertex; click a point to move there

The centre vertex and its neighbour rings are highlighted everywhere: compact on the
sphere, but split across the seams of the net, where every stored copy is marked.

Controls on the left: base solid, colouring (parent face of the base solid, position, or
spherical cell area relative to the mean -- compare the tetrahedron with the others),
centre (a base-solid corner, a seam vertex or an interior one), frequency and number of
rings.  "Map viewer" opens the dome in the rotatable ProjectionViewer (drag the map).

Needs: numpy, matplotlib with a GUI backend (in Jupyter use the notebook version,
examples/notebooks/11_interactive_base_polyhedra.ipynb).
"""
import argparse

import dome_utils  # noqa: F401  (adds the repo root to sys.path when run from a checkout)
import matplotlib.pyplot as plt
import numpy as np
from base_explorer import BASES, COLOURINGS, BaseExplorer
from matplotlib.widgets import Button, RadioButtons, Slider

from mt.geodesicdome.interactive import ProjectionViewer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--base', default='icosahedron',
                    help='tetrahedron | icosahedron | dodecahedron (or tetra / icosa / dodeca)')
parser.add_argument('--freq', type=int, default=4, help='dome frequency (default 4)')
parser.add_argument('--rings', type=int, default=3, help='neighbour rings to highlight (default 3)')
parser.add_argument('--colouring', default='base face', choices=COLOURINGS)
parser.add_argument('--centre', default='interior', choices=['interior', 'seam', 'corner'])
parser.add_argument('--save', metavar='PATH', help='save the figure to an image instead of opening a window')
args = parser.parse_args()

fig = plt.figure(figsize=(17, 6.6))
fig.canvas.manager and fig.canvas.manager.set_window_title('mt.geodesicdome: geodesic domes on three base solids')
explorer = BaseExplorer(fig, base=args.base, frequency=args.freq, rings=args.rings,
                        colouring=args.colouring, rect=(0.12, 0.0, 0.88, 1.0))
explorer.select_kind(args.centre)

if args.save:
    fig.savefig(args.save, dpi=100)
    print(f'saved {args.save}')
    raise SystemExit

# --- controls (left column; the explorer redraws only its own three panels)
fig.text(0.012, 0.93, 'base solid', fontsize=9, weight='bold')
base = RadioButtons(fig.add_axes([0.01, 0.76, 0.10, 0.16], frameon=False), BASES, active=BASES.index(explorer.base))
base.on_clicked(explorer.set_base)

fig.text(0.012, 0.72, 'colouring', fontsize=9, weight='bold')
colouring = RadioButtons(fig.add_axes([0.01, 0.56, 0.10, 0.15], frameon=False), COLOURINGS,
                         active=COLOURINGS.index(explorer.colouring))
colouring.on_clicked(explorer.set_colouring)

fig.text(0.012, 0.52, 'centre on', fontsize=9, weight='bold')
centre_buttons = []
for i, kind in enumerate(('interior', 'seam', 'corner')):
    button = Button(fig.add_axes([0.01 + i * 0.034, 0.46, 0.032, 0.045]), kind)
    button.label.set_fontsize(7)
    button.on_clicked(lambda _event, kind=kind: explorer.select_kind(kind))
    centre_buttons.append(button)                              # keep references alive

freq = Slider(fig.add_axes([0.035, 0.36, 0.07, 0.03]), 'freq', 1, 12, valinit=explorer.frequency, valstep=1)
freq.on_changed(lambda value: explorer.set_frequency(int(value)))
rings = Slider(fig.add_axes([0.035, 0.30, 0.07, 0.03]), 'rings', 0, 8, valinit=explorer.rings, valstep=1)
rings.on_changed(lambda value: explorer.set_rings(int(value)))

viewers = []                                                   # keep references to opened viewers


def open_viewer(_event):
    """The current dome in the rotatable map viewer, centred on the current vertex."""
    viewer = ProjectionViewer(explorer.dome, colors='base',
                              title=f"GeodesicDome({explorer.frequency}, base='{explorer.base}') — drag the map")
    lat, lon = np.degrees(explorer._latlon(explorer.centre.coord))
    viewer.set_view(lat, lon)
    viewer.fig.show()
    viewers.append(viewer)


viewer_button = Button(fig.add_axes([0.01, 0.20, 0.10, 0.05]), 'Map viewer')
viewer_button.on_clicked(open_viewer)
fig.text(0.012, 0.12, 'click the net or the map\nto choose the centre;\ndrag the sphere to turn it',
         fontsize=8, color='#555555')

explorer.connect()
plt.show()
