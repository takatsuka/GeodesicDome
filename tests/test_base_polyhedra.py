# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""GeodesicDome on the tetrahedron, icosahedron and dodecahedron (the ``base`` argument)."""

import copy
import pickle
from collections import deque

import numpy as np
import pytest

from mt.geodesicdome.grid.geodesicdome import (
    DodecahedronDome,
    GeodesicDome,
    IcosahedronDome,
    NetDome,
    TetrahedronDome,
)
from mt.geodesicdome.grid.polyhedra import BASES, normalise_base
from mt.geodesicdome.interactive.mesh import unique_mesh

# base -> (k in k f^2 + 2 points, triangles per f^2, number of corners, their degree)
SPEC = {'tetrahedron': (2, 4, 4, 3), 'icosahedron': (10, 20, 12, 5), 'dodecahedron': (30, 60, 12, 5)}
FREQS = [1, 2, 3, 5]


def mesh_graph(dome):
    points, faces, index_map = unique_mesh(dome)
    nbrs = [set() for _ in range(len(points))]
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            nbrs[a].add(b)
            nbrs[b].add(a)
    return points, faces, index_map, nbrs


@pytest.mark.parametrize('name', ['tetra', 'Tetrahedron', 'tet', 4, 'icosa', 'ico', None, 20,
                                  'dodeca', 'DODECAHEDRON', 'dod', 12])
def test_aliases(name):
    dome = GeodesicDome(1, base=name)
    assert dome.base == normalise_base(name)
    assert isinstance(dome, GeodesicDome)


def test_dispatch_and_repr():
    assert type(GeodesicDome(2)) is IcosahedronDome
    assert type(GeodesicDome(2, base='tetra')) is TetrahedronDome
    assert type(GeodesicDome(2, base='dodeca')) is DodecahedronDome
    assert type(GeodesicDome(frequency=2, base='dodeca')) is DodecahedronDome
    assert repr(GeodesicDome(4)) == 'GeodesicDome(frequency=4)'
    assert repr(GeodesicDome(4, 'tetra')) == "GeodesicDome(frequency=4, base='tetrahedron')"
    assert TetrahedronDome(3).frequency == 3
    with pytest.raises(ValueError):
        GeodesicDome(2, base='cube')
    with pytest.raises(ValueError):
        TetrahedronDome(2, base='icosa')


@pytest.mark.parametrize('base', BASES)
@pytest.mark.parametrize('f', FREQS)
def test_sizes_and_orientation(base, f):
    k, t, n_corners, corner_degree = SPEC[base]
    dome = GeodesicDome(f, base=base)
    xyz = dome.get_all_xyz()
    points, faces, _, nbrs = mesh_graph(dome)
    assert len(points) == k * f * f + 2
    assert len(faces) == t * f * f
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0)
    p = points[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.einsum('ij,ij->i', normal, p.sum(axis=1)) > 0).all()        # CCW from outside
    degree = np.array([len(s) for s in nbrs])
    assert (degree == corner_degree).sum() == n_corners
    assert set(degree) == {corner_degree, 6} or (f == 1 and set(degree) <= {corner_degree, 6})
    # Euler characteristic of a sphere
    assert len(points) - sum(degree) // 2 + len(faces) == 2


def test_tetrahedron_is_the_4hsom_orthogonal_array():
    for f in (1, 2, 3, 6):
        dome = GeodesicDome(f, base='tetra')
        assert (dome.x_max, dome.y_max) == (2 * f, f)
        assert len(dome.get_all_vertices()) == (f + 1) * (2 * f + 1)          # every cell is used
        for y in range(f + 1):
            left, right = dome.get_vertex_at(0, y), dome.get_vertex_at(2 * f, y)
            assert right in left.same_vertices
        for i in range(f):
            for y in (0, f):
                a, b = dome.get_vertex_at(i, y), dome.get_vertex_at(2 * f - i, y)
                np.testing.assert_allclose(a.coord, b.coord, atol=1e-12)
    # the 3-frequency array of de Sousa & Oliveira: 4 x 7 cells, 20 neurons + 8 copies
    dome = GeodesicDome(3, base='tetra')
    assert len(dome.get_all_vertices()) == 28
    assert len(unique_mesh(dome)[0]) == 20
    np.testing.assert_allclose(dome.get_vertex_at(3, 3).coord, [0, 1, 0], atol=1e-12)   # D: north pole


@pytest.mark.parametrize('base', BASES)
@pytest.mark.parametrize('f', [1, 2, 4])
def test_index_neighbours_match_mesh(base, f):
    """get_neighbours (grid offsets + seam copies) gives exactly the mesh neighbours of every vertex."""
    dome = GeodesicDome(f, base=base)
    _, _, index_map, nbrs = mesh_graph(dome)
    for v in dome.get_all_vertices():
        dome.unmark_vertices()
        found = [index_map[n.id] for n in dome.get_neighbours(v, False)]
        assert len(found) == len(set(found)), 'a neighbour was returned twice'
        assert set(found) == nbrs[index_map[v.id]]


@pytest.mark.parametrize('base', BASES)
def test_rings_match_breadth_first_search(base):
    dome = GeodesicDome(6, base=base)
    _, _, index_map, nbrs = mesh_graph(dome)
    rng = np.random.default_rng(0)
    vertices = dome.get_all_vertices()
    for idx in rng.choice(len(vertices), 25, replace=False):
        v = vertices[idx]
        dome.unmark_vertices()
        rings = dome.get_neighbours_in_distance(v, 4)
        start = index_map[v.id]
        dist = {start: 0}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b in nbrs[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    queue.append(b)
        for level, ring in enumerate(rings, start=1):
            got = [index_map[n.id] for n in ring]
            assert len(got) == len(set(got))
            assert set(got) == {p for p, d in dist.items() if d == level}


def test_interior_rings():
    dome = GeodesicDome(8, base='dodeca')
    v = dome.get_vertex_at(20, 30)
    dome.unmark_vertices()
    assert [len(r) for r in dome.get_neighbours_in_distance(v, 3)] == [6, 12, 18]


@pytest.mark.parametrize('base', ['tetrahedron', 'dodecahedron'])
def test_split_is_cumulative(base):
    dome = GeodesicDome(2, base=base)
    dome.split(3)
    ref = GeodesicDome(6, base=base)
    assert dome.frequency == 6
    np.testing.assert_allclose(dome.get_all_xyz(), ref.get_all_xyz())


def test_generic_icosahedron_net_matches_icosahedron_dome():
    """The generic grid engine on the icosahedron net gives the same sphere as IcosahedronDome."""

    class _GenericIcosahedron(NetDome):
        base = 'icosahedron'

    for f in (1, 2, 5):
        a, b = unique_mesh(IcosahedronDome(f)), unique_mesh(_GenericIcosahedron(f))
        assert len(a[0]) == len(b[0])
        cos = a[0] @ b[0].T
        match = cos.argmax(axis=1)                            # point of b for every point of a
        np.testing.assert_allclose(cos.max(axis=1), 1.0, atol=1e-9)   # IcosahedronDome rounds its latitude
        assert len(set(match)) == len(match)
        sides = ((0, 1), (1, 2), (2, 0))
        edges_a = {tuple(sorted((match[t[i]], match[t[j]]))) for t in a[1] for i, j in sides}
        edges_b = {tuple(sorted((t[i], t[j]))) for t in b[1] for i, j in sides}
        assert edges_a == edges_b

def test_dodecahedron_geometry():
    dome = GeodesicDome(1, base='dodeca')
    points, _, _, nbrs = mesh_graph(dome)
    degree = np.array([len(s) for s in nbrs])
    centres, corners = points[degree == 5], points[degree == 6]
    # dodecahedron corners: 3 nearest corners each at the dodecahedron's edge angle
    cos = corners @ corners.T
    np.fill_diagonal(cos, -2)
    edge = np.arccos(np.sort(cos, axis=1)[:, -1])
    np.testing.assert_allclose(edge, edge[0])
    np.testing.assert_allclose(edge[0], np.arccos(np.sqrt(5) / 3), atol=1e-12)
    # pentagon centres point like the icosahedron's corners
    ico = unique_mesh(IcosahedronDome(1))[0]
    match = np.abs(centres @ ico.T).max(axis=1)
    np.testing.assert_allclose(match, 1.0, atol=1e-5)


@pytest.mark.parametrize('base', BASES)
def test_nets_do_not_overlap(base):
    """No two small triangles of the net occupy the same cell of the index grid."""
    dome = GeodesicDome(4, base=base)
    cells = [frozenset((v.x, v.y) for v in tri) for tri in zip(*[iter(dome.get_faces())] * 3, strict=True)]
    assert len(cells) == len(set(cells))
    assert all(len(c) == 3 for c in cells)

@pytest.mark.parametrize('base', BASES)
def test_copy_and_pickle(base):
    dome = GeodesicDome(2, base=base)
    for clone in (copy.deepcopy(dome), pickle.loads(pickle.dumps(dome))):
        assert type(clone) is type(dome)
        np.testing.assert_allclose(clone.get_all_xyz(), dome.get_all_xyz())
