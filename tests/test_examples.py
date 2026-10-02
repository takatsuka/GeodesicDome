# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""Smoke test for examples/base_explorer.py (used by example 11 and its notebook); Agg backend, no window."""
import os
import sys
from types import SimpleNamespace

import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from base_explorer import COLOURINGS, BaseExplorer  # noqa: E402


@pytest.mark.parametrize('base', ['tetrahedron', 'icosahedron', 'dodecahedron'])
def test_base_explorer(base):
    fig = plt.figure(figsize=(12, 4.5))
    explorer = BaseExplorer(fig, base=base, frequency=3, rings=2)
    n_axes = len(fig.axes)
    for kind in ('corner', 'seam', 'interior'):
        explorer.select_kind(kind)
    for colouring in COLOURINGS[:2]:
        explorer.set_colouring(colouring)
    explorer.set_frequency(2)
    explorer.set_rings(3)
    assert len(fig.axes) == n_axes                       # redraws replace the panels, nothing piles up
    i = len(explorer.vertices) // 3
    explorer._on_click(SimpleNamespace(inaxes=explorer.axnet, xdata=explorer.net[i, 0], ydata=explorer.net[i, 1]))
    assert explorer.centre is explorer.vertices[i]
    explorer._on_click(SimpleNamespace(inaxes=explorer.axmap, xdata=0.0, ydata=0.0))   # the map centre
    assert explorer.centre.coord @ explorer.vertices[i].coord > 0.99
    explorer.set_colouring('cell area')
    assert len(fig.axes) == n_axes + 1                   # + colour bar
    plt.close(fig)
