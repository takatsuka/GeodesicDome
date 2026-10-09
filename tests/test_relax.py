# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""Lloyd relaxation: the grid stays, the cells even out, the seams stay together."""
import numpy as np
import pytest

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.relax import CONVERGED, cell_areas, lloyd, outward

BASES = [('icosahedron', 8), ('tetrahedron', 8), ('dodecahedron', 4)]


def unique_points(dome):
    cls = dome.point_index
    x = np.empty((dome.n_points, 3))
    x[cls] = dome.get_all_xyz()
    return x, outward(x, cls[dome.get_all_triangles().reshape(-1, 3)])


def area_cv(dome):
    a = cell_areas(*unique_points(dome))
    return a.std() / a.mean()


@pytest.mark.parametrize('base,f', BASES)
def test_relax_keeps_grid_and_evens_cells(base, f):
    plain = GeodesicDome(f, base=base)
    dome = GeodesicDome(f, base=base, relax=True)
    assert dome.relaxed and not plain.relaxed
    assert 0 < dome.relax_steps < CONVERGED['iters']               # converged before the step limit
    # same grid: positions, faces, seam classes and neighbours
    assert np.array_equal(dome._xy, plain._xy)
    assert np.array_equal(dome.get_all_triangles(), plain.get_all_triangles())
    assert np.array_equal(dome.point_index, plain.point_index)
    for i in range(0, len(dome._coords), 7):
        assert dome.within_hops(i, 2) == plain.within_hops(i, 2)
    xyz = dome.get_all_xyz()
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0)
    # seam copies stay one point
    for i, others in dome._same.items():
        for j in others:
            assert np.array_equal(xyz[i], xyz[j])
    # the points moved and the cells are more even
    assert not np.allclose(xyz, plain.get_all_xyz())
    assert area_cv(dome) < 0.6 * area_cv(plain)


@pytest.mark.parametrize('base,f', BASES)
def test_relaxed_mesh_is_delaunay(base, f):
    """Fixed connectivity is sound: no triangle folds over and every triangle's circumcircle is empty."""
    x, tri = unique_points(GeodesicDome(f, base=base, relax=True))
    a, b, c = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    normal = np.cross(b - a, c - a)
    assert (np.einsum('ij,ij->i', normal, a) > 0).all()
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # on the sphere, the circumcircle is the plane through a, b, c; it is empty if all points lie on one side
    side = x @ normal.T - np.einsum('ij,ij->i', normal, a)[None, :]
    assert (side <= 1e-12).all()


def test_icosahedral_cell_areas_match_spiral_comparison():
    """docs/lattice_comparison.md in the spiral repository: cell-area CV 0.132 -> 0.039 at N = 2 562."""
    plain, dome = GeodesicDome(16), GeodesicDome(16, relax=True)
    assert dome.n_points == 2562
    assert area_cv(plain) == pytest.approx(0.132, abs=5e-4)
    assert area_cv(dome) == pytest.approx(0.039, abs=5e-4)


def test_relax_method_settings_and_in_place_update():
    dome = GeodesicDome(4)
    moved = np.linalg.norm(GeodesicDome(4, relax=True).get_all_xyz() - dome.get_all_xyz(), axis=1)
    v = dome.get_all_vertices()[int(moved.argmax())]           # the point that moves most
    before = v.coord.copy()
    ll = v.latlon_coord.copy()
    assert dome.relax(iters=3, omega=1.0) is dome
    assert dome.relax_steps == 3
    assert np.abs(v.coord - before).max() > 1e-3                # vertex objects see the new coordinates
    assert np.array_equal(v.coord, dome.get_all_xyz()[v.id])
    assert np.abs(v.latlon_coord - ll).max() > 1e-4             # cached lat/lon recomputed
    assert repr(dome) == 'GeodesicDome(frequency=4, relaxed=True)'


def test_relax_option_dict_and_errors():
    a = GeodesicDome(4, relax={'iters': 5, 'omega': 1.0})
    assert a.relax_steps == 5
    b = GeodesicDome(4)
    b.relax(iters=5, omega=1.0)
    assert np.array_equal(a.get_all_xyz(), b.get_all_xyz())
    with pytest.raises(TypeError):
        GeodesicDome(4, relax='yes')
    with pytest.raises(ValueError):
        GeodesicDome(4).relax(omega=2.0)


def test_split_resets_relaxed():
    dome = GeodesicDome(2, relax=True)
    dome.split(2)
    assert not dome.relaxed and dome.relax_steps == 0


def test_netdome_arc_length_follows_relaxation():
    plain = GeodesicDome(4, base='dodecahedron')
    dome = GeodesicDome(4, base='dodecahedron', relax=True)
    assert dome.arcLength != plain.arcLength
    assert dome.arcLength == pytest.approx(plain.arcLength, rel=0.05)


def test_lloyd_fixed_point():
    """A converged mesh is its own centroid for any omega."""
    x, tri = unique_points(GeodesicDome(4, relax=True))
    for omega in (1.0, 1.8):
        assert np.abs(lloyd(x, tri, iters=1, omega=omega) - x).max() < 1e-6


# ---------------------------------------------------------------------------------- backends
class _Float32Device:
    """A stand-in for the Apple GPU: NumPy in float32, flagged as a GPU."""

    @staticmethod
    def make():
        from mt.geodesicdome.backend import _ArrayModuleBackend

        class Float32(_ArrayModuleBackend):
            name = 'numpy'

            def __init__(self):
                super().__init__(np, 'fake-gpu', np.float32, True, 1, 1 << 28, 1 << 28)

        return Float32()


def test_numpy_backend_matches_reference_step():
    """The batched step equals the per-corner loop of the spiral repository's relax.py."""
    from mt.geodesicdome.relax import _unit, triangle_areas

    x, tri = unique_points(GeodesicDome(4))
    f = len(tri)
    idx = np.arange(f)
    num, den = np.zeros_like(x), np.zeros(len(x))
    a, b, c = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    cc = _unit(np.cross(b - a, c - a))
    for k in range(3):
        v = tri[:, k]
        p = x[v]
        for m in ((k + 1) % 3, (k + 2) % 3):
            mid = _unit(p + x[tri[:, m]])
            area = triangle_areas(np.vstack([p, cc, mid]), np.c_[idx, f + idx, 2 * f + idx])
            np.add.at(num, v, area[:, None] * _unit(p + cc + mid))
            np.add.at(den, v, area)
    expected = _unit(x + 1.8 * (_unit(num / den[:, None]) - x))
    assert np.abs(lloyd(x, tri, iters=1, omega=1.8, backend='numpy') - expected).max() < 1e-14


def test_float32_device_finishes_in_float64():
    """On a float32 device the run stops at its precision floor and the CPU finishes to tol."""
    x, tri = unique_points(GeodesicDome(8))
    ref = lloyd(x, tri, backend='numpy', **CONVERGED)
    got, steps = lloyd(x, tri, backend=_Float32Device.make(), return_steps=True, **CONVERGED)
    assert got.dtype == np.float64
    assert steps < CONVERGED['iters']
    assert np.abs(got - ref).max() < 1e-5
    # converged in float64: one more step moves nothing beyond tol
    assert np.abs(lloyd(got, tri, iters=1, omega=1.8, backend='numpy') - got).max() < 1e-6


def test_dome_relax_backend_argument():
    a = GeodesicDome(8, relax={'backend': 'cpu'})
    b = GeodesicDome(8).relax(backend=_Float32Device.make())
    assert np.abs(a.get_all_xyz() - b.get_all_xyz()).max() < 1e-5


def test_torch_backend_if_installed():
    torch = pytest.importorskip('torch')
    x, tri = unique_points(GeodesicDome(8))
    ref = lloyd(x, tri, backend='numpy', **CONVERGED)
    got = lloyd(x, tri, backend='torch:cpu', **CONVERGED)
    assert np.abs(got - ref).max() < 1e-9
    if torch.cuda.is_available() or (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
        gpu = lloyd(x, tri, backend='gpu', **CONVERGED)
        assert np.abs(gpu - ref).max() < 1e-5
