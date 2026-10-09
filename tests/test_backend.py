# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import pickle

import numpy as np
import pytest

from mt.geodesicdome import backend, compute
from mt.geodesicdome.backend import NumpyBackend, _ArrayModuleBackend, get_backend
from mt.geodesicdome.grid.geodesicdome import GeodesicDome


def _cpu_backends():
    """Every backend that can run on this machine, plus a numpy-module stand-in for cupy."""
    found = [NumpyBackend(threads=1), NumpyBackend(threads=4)]
    found.append(_ArrayModuleBackend(np, 'cpu', np.float64, False, 1, 1 << 16, 1 << 20))   # tiny blocks
    for spec in backend.available_backends():
        if spec != 'numpy':
            found.append(get_backend(spec))
    if 'torch:cpu' in backend.available_backends():            # single precision, as on a GPU
        found.append(get_backend('torch:cpu', dtype='float32'))
    return found


BACKENDS = _cpu_backends()
IDS = [repr(b) for b in BACKENDS]


def test_auto_backend_and_describe():
    b = get_backend()
    assert b is get_backend()                      # cached
    assert b.spec.split(':')[0] in ('torch', 'cupy', 'numpy')
    assert 'default:' in backend.describe()
    assert 'numpy' in backend.available_backends()


def test_explicit_and_bad_specs(monkeypatch):
    assert get_backend('numpy').name == 'numpy'
    assert get_backend('cpu').device == 'cpu'
    with pytest.raises(ValueError):
        get_backend('abacus')
    monkeypatch.setenv(backend.ENV_MEMORY, '1MB')
    assert NumpyBackend().block_bytes == 1 << 20


def test_default_can_be_set():
    old = get_backend()
    try:
        assert backend.set_default_backend('numpy').name == 'numpy'
        assert get_backend().name == 'numpy'
    finally:
        backend.set_default_backend(old)


def test_backend_pickles_as_spec():
    b = get_backend('numpy')
    assert pickle.loads(pickle.dumps(b)).name == 'numpy'


@pytest.mark.parametrize('b', BACKENDS, ids=IDS)
def test_operations_agree_with_numpy(b):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 7))
    ad = b.asarray(a)
    assert np.allclose(b.to_numpy(b.exp(ad)), np.exp(a), rtol=1e-5)
    assert np.array_equal(b.to_numpy(b.argmin(ad, axis=1)), a.argmin(axis=1))
    assert np.array_equal(b.to_numpy(b.smallest(ad, 2)), np.argsort(a, axis=1)[:, :2])
    assert np.array_equal(b.to_numpy(b.largest(ad, 3)), np.argsort(-a, axis=1)[:, :3])
    idx = rng.integers(0, 5, size=50)
    target = b.index_add(b.zeros((5, 7)), b.as_index(idx), ad)
    expected = np.zeros((5, 7))
    np.add.at(expected, idx, a)
    assert np.allclose(b.to_numpy(target), expected, rtol=1e-5)
    out = b.map_blocks(lambda s: b.to_numpy(ad[s]).sum(), 50, 7)
    assert len(out) == 8 and np.isclose(sum(out), a.sum())


@pytest.mark.parametrize('b', BACKENDS, ids=IDS)
def test_angular_distance_and_nearest_vertex(b):
    mesh = compute.DomeArrays.from_dome(GeodesicDome(6))
    p = mesh.points
    expected = np.arccos(np.clip(p @ p.T, -1, 1))
    got = compute.angular_distance(p, backend=b)
    assert got.shape == (mesh.n, mesh.n)
    assert np.allclose(got, expected, atol=2e-3 if b.dtype == np.float32 else 1e-9)
    rings = compute.ring_distance(mesh, backend=b)
    assert np.allclose(rings[mesh.edges[:, 0], mesh.edges[:, 1]], 1.0, atol=0.25)

    rng = np.random.default_rng(1)
    q = rng.normal(size=(300, 3)) * 5.0
    idx, ang = compute.nearest_vertex(p, q, backend=b, return_angle=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    assert np.array_equal(idx, np.argmax(qn @ p.T, axis=1))
    assert np.allclose(ang, np.arccos(np.clip((qn * p[idx]).sum(1), -1, 1)), atol=1e-3)
    two = compute.nearest_vertex(p, q, k=2, backend=b)
    assert two.shape == (300, 2) and np.array_equal(two[:, 0], idx)
    assert compute.nearest_vertex(p, q[0], backend=b).shape == ()


def test_dome_arrays():
    mesh = compute.DomeArrays.from_dome(GeodesicDome(4))
    assert mesh.n == 162 and len(mesh.faces) == 320 and len(mesh.edges) == 480
    assert 0 < mesh.ring_length < 0.31
