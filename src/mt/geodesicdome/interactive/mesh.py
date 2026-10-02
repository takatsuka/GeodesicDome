# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Mesh helpers: turn a Manifold (e.g. GeodesicDome) into a clean triangle mesh.
"""

import numpy as np


def unique_mesh(manifold, decimals: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapses the duplicated seam vertices of a GeodesicDome into a clean triangle mesh.

    :param manifold: a GeodesicDome (or any Manifold with triangular faces)
    :param decimals: rounding used to decide that two stored vertices are the same point
    :return: (points, faces, index_map)
             points    -- (V, 3) unique coordinates
             faces     -- (F, 3) triangle indices into ``points``
             index_map -- (N,) for every stored vertex id, its row in ``points``
    """
    if manifold.get_number_of_vertices_per_face() != 3:
        raise ValueError('unique_mesh needs a manifold with triangular faces')
    triangles = manifold.get_all_triangles().reshape(-1, 3)   # also assigns vertex.id
    xyz = manifold.get_all_xyz()
    points, index_map = np.unique(np.round(xyz, decimals), axis=0, return_inverse=True)
    index_map = index_map.reshape(-1)
    return points, index_map[triangles], index_map
