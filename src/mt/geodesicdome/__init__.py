# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.geodesicdome -- geodesic domes, grids and map projections.

Subpackages
    mt.geodesicdome.grid          GeodesicDome (sphere on an icosahedron, tetrahedron or dodecahedron) and Plane grids
    mt.geodesicdome.projection    map projections (Equal Earth, Kavrayskiy VII, Wagner VI/III)
    mt.geodesicdome.interactive   rotatable interactive map viewer (needs matplotlib)
    mt.geodesicdome.backend       compute backends: GPU (CUDA, Apple MPS) when available, else all CPU cores
    mt.geodesicdome.compute       array kernels on domes (great-circle distances, nearest vertex) on that backend

The indexed geodesic data structure is described in
Y. Wu and M. Takatsuka, "Spherical self-organizing map using efficient indexed
geodesic data structure", Neural Networks 19(6-7):900-910, 2006.
doi:10.1016/j.neunet.2006.05.021

The tetrahedral dome uses the orthogonal-array layout of R. M. de Sousa and R. C. L. Oliveira,
"Optimization of geodesic self-organizing map using tessellated tetrahedron as spherical
lattice", IJCNN 2012.

Note: `mt` itself is a namespace package (no __init__.py), so other
mt.* distributions can be installed alongside this one.
"""

__version__ = '1.3.1'
