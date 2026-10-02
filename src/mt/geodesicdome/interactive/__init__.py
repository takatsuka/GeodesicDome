# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.geodesicdome.interactive -- rotate a geodesic dome interactively inside a map projection.

    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    from mt.geodesicdome.interactive import ProjectionViewer

    viewer = ProjectionViewer(GeodesicDome(8), 'Equal Earth')
    viewer.show()            # drag on the map to rotate the sphere

Contents
    ProjectionViewer   the interactive matplotlib viewer          (needs matplotlib)
    SphereMap          seam- and pole-correct projection of a rotated mesh (numpy only)
    unique_mesh        GeodesicDome -> de-duplicated (points, faces, index_map)
    rotation           small 3x3 rotation helpers (view_rotation, drag_rotation, ...)
"""
from mt.geodesicdome.interactive import rotation
from mt.geodesicdome.interactive.mesh import unique_mesh
from mt.geodesicdome.interactive.sphere_map import SphereMap

__all__ = ['ProjectionViewer', 'SphereMap', 'unique_mesh', 'rotation']


def __getattr__(name):
    # import the matplotlib-based viewer lazily, so SphereMap works without matplotlib
    if name == 'ProjectionViewer':
        from mt.geodesicdome.interactive.viewer import ProjectionViewer
        return ProjectionViewer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
