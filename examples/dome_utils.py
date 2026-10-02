# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Small helpers shared by the example scripts.

GeodesicDome stores its vertices on an "unfolded" rectilinear net, so a point that
lies on a seam of the net (for example the 12 icosahedron corners) is stored more
than once -- each copy is listed in ``vertex.same_vertices``.  That is exactly what
makes fast (x, y) neighbour look-ups possible, but for rendering / exporting a clean
mesh you usually want every physical point only once.  ``unique_mesh`` does that.
"""
import os
import sys

import numpy as np

# Allow running the examples straight from a source checkout without `pip install`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt.geodesicdome.grid.geodesicdome import GeodesicDome, GeodesicVertex  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def output_path(filename: str) -> str:
    """Returns examples/output/<filename>, creating the folder if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


def unique_mesh(dome: GeodesicDome, decimals: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapses the duplicated seam vertices of a dome into a clean triangle mesh.

    :param dome: a GeodesicDome
    :param decimals: rounding used to decide that two stored vertices are the same point
    :return: (points, faces, index_map)
             points    -- (V, 3) unique xyz coordinates on the unit sphere, V = 10 f^2 + 2
             faces     -- (F, 3) triangle indices into ``points``,        F = 20 f^2
             index_map -- (N,)  for every stored vertex id, its index into ``points``
    """
    xyz = dome.get_all_xyz()                                # (N, 3), N includes seam copies
    triangles = dome.get_all_triangles().reshape(-1, 3)     # indices into xyz (vertex.id)
    points, index_map = np.unique(np.round(xyz, decimals), axis=0, return_inverse=True)
    index_map = index_map.reshape(-1)
    return points, index_map[triangles], index_map


def neighbour_rings(dome: GeodesicDome, vertex: GeodesicVertex, distance: int) -> list[list[GeodesicVertex]]:
    """
    Convenience wrapper: clears the 'visited' flags and returns the neighbour rings
    (ring 1 = immediate neighbours, ring 2 = neighbours of those, ...).
    """
    dome.unmark_vertices()
    rings = dome.get_neighbours_in_distance(vertex, distance)
    dome.unmark_vertices()
    return rings


def shade(face_normals: np.ndarray, light=(0.4, 0.5, 0.8)) -> np.ndarray:
    """Simple Lambert shading factor in [0.35, 1] for a set of face normals."""
    light = np.asarray(light, dtype=float)
    light /= np.linalg.norm(light)
    lambert = np.clip(face_normals @ light, 0.0, 1.0)
    return 0.35 + 0.65 * lambert
