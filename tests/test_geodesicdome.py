# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""Core GeodesicDome invariants (the numbers in the README's frequency table)."""

import numpy as np
import pytest

from mt.geodesicdome.grid.geodesicdome import GeodesicDome


@pytest.mark.parametrize(
    ("frequency", "stored_vertices"),
    [(1, 22), (2, 63), (4, 205), (8, 729), (12, 1573)],
)
def test_sizes(frequency, stored_vertices):
    dome = GeodesicDome(frequency)
    xyz = dome.get_all_xyz()
    triangles = dome.get_all_triangles().reshape(-1, 3)
    unique_points = np.unique(np.round(xyz, 9), axis=0)

    assert len(unique_points) == 10 * frequency**2 + 2
    assert len(triangles) == 20 * frequency**2
    assert len(xyz) == stored_vertices  # includes seam copies
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0)


def test_split_multiplies_frequency():
    dome = GeodesicDome(2)
    dome.split(3)
    assert dome.frequency == 6


def test_neighbour_rings():
    dome = GeodesicDome(8)
    v = dome.get_vertex_at(12, 14)
    dome.unmark_vertices()
    assert len(dome.get_neighbours(v, False)) == 6
    dome.unmark_vertices()
    assert [len(r) for r in dome.get_neighbours_in_distance(v, 3)] == [6, 12, 18]
