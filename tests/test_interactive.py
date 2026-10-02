# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Tests for mt.geodesicdome.interactive.  Run with  `pytest`  or  `python tests/test_interactive.py`.
Needs numpy and matplotlib (no window is opened: the Agg backend is used).
"""
import pytest

matplotlib = pytest.importorskip('matplotlib')  # the core library does not need it; skip if absent
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from mt.geodesicdome.grid.geodesicdome import GeodesicDome  # noqa: E402
from mt.geodesicdome.interactive import ProjectionViewer, SphereMap  # noqa: E402
from mt.geodesicdome.interactive import rotation as rot  # noqa: E402
from mt.geodesicdome.interactive.viewer import DEFAULT_PROJECTIONS  # noqa: E402


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = [np.eye(3)]
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))              # uniformly random orthogonal matrix
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]                   # make it a proper rotation (no mirror)
        out.append(q)
    # symmetric views that put the pole on edges / vertices of the mesh
    out += [rot.view_rotation(np.radians(lat), 0.0) for lat in (90, 165, 172.5)]
    return out


def _uncovered_pixels(sphere_map, rotation):
    """Renders the faces and counts background pixels left inside the (slightly shrunk) outline."""
    outline = sphere_map.outline()
    centre = outline.mean(axis=0)
    inner = centre + 0.98 * (outline - centre)

    def render(draw_faces):
        fig = plt.figure(figsize=(6, 3), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(outline[:, 0].min(), outline[:, 0].max())
        ax.set_ylim(outline[:, 1].min(), outline[:, 1].max())
        if draw_faces:
            clip = Polygon(outline, closed=True, facecolor='none', edgecolor='none')
            ax.add_patch(clip)
            polygons, _ = sphere_map.polygons(rotation)
            faces = PolyCollection(polygons, facecolors='black', edgecolors='none', antialiased=False)
            faces.set_clip_path(clip)
            ax.add_collection(faces)
        else:
            ax.add_patch(Polygon(inner, closed=True, facecolor='black', edgecolor='none', antialiased=False))
        fig.canvas.draw()
        dark = np.asarray(fig.canvas.buffer_rgba())[:, :, 0] < 128
        plt.close(fig)
        return dark

    return int((render(False) & ~render(True)).sum())


def test_polygons_cover_every_face():
    sphere_map = SphereMap.from_manifold(GeodesicDome(4), DEFAULT_PROJECTIONS['Equal Earth'])
    for r in _random_rotations(10):
        polygons, face_index = sphere_map.polygons(r)
        assert polygons.shape[1:] == (6, 2)
        assert len(polygons) == len(face_index)
        assert np.isfinite(polygons).all()
        assert set(face_index) == set(range(len(sphere_map.faces)))


def test_no_gaps_in_any_projection():
    dome = GeodesicDome(5)
    for name, projection in DEFAULT_PROJECTIONS.items():
        sphere_map = SphereMap.from_manifold(dome, projection)
        for r in _random_rotations(6, seed=1):
            assert _uncovered_pixels(sphere_map, r) == 0, name


def test_view_rotation_round_trip():
    for lat, lon in ((0, 0), (35, -120), (-89, 45), (60, 179)):
        r = rot.view_rotation(np.radians(lat), np.radians(lon))
        assert np.allclose(np.degrees(rot.centre_of_view(r)), (lat, lon), atol=1e-9)


def test_drag_rotates_like_the_map():
    viewer = ProjectionViewer(GeodesicDome(4))
    canvas, ax = viewer.fig.canvas, viewer.ax
    width, height = viewer._map_size

    def drag(dx, dy):
        x0, y0 = ax.transData.transform((0, 0))
        x1, y1 = ax.transData.transform((dx, dy))
        canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', canvas, x0, y0, button=1))
        for t in np.linspace(0, 1, 8)[1:]:
            canvas.callbacks.process('motion_notify_event',
                                     MouseEvent('motion_notify_event', canvas,
                                                x0 + t * (x1 - x0), y0 + t * (y1 - y0), button=1))
        canvas.callbacks.process('button_release_event',
                                 MouseEvent('button_release_event', canvas, x1, y1, button=1))

    drag(width / 4, 0)                       # a quarter of the map to the right = 90 degrees
    assert np.allclose(viewer.centre, (0, -90), atol=0.5)
    viewer.reset()
    drag(0, height / 6)                      # a sixth of the height upwards = 30 degrees
    assert np.allclose(viewer.centre, (-30, 0), atol=0.5)
    plt.close(viewer.fig)


def test_viewer_api():
    dome = GeodesicDome(3)
    values = dome.get_all_xyz()[:, 2]
    viewer = ProjectionViewer(dome, 'Wagner VI', colors=values, view=(20, 40))
    assert np.allclose(viewer.centre, (20, 40))
    viewer.rotate(d_lon=10)
    viewer.set_projection('Kavrayskiy VII')
    assert viewer.projection_name == 'Kavrayskiy VII'
    viewer.set_colors('icosahedron')
    viewer.set_view(-33.9, 151.2)
    assert np.allclose(viewer.centre, (-33.9, 151.2))
    try:
        viewer.set_rotation(np.diag([1.0, 1.0, -1.0]))     # a mirror image is not a rotation
        raise AssertionError('reflection accepted')
    except ValueError:
        pass
    plt.close(viewer.fig)


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(f'{name}: ok')


@pytest.mark.parametrize(('base', 'n_parents'), [('tetrahedron', 4), ('icosahedron', 20), ('dodecahedron', 12)])
def test_viewer_on_every_base(base, n_parents):
    dome = GeodesicDome(4, base=base)
    viewer = ProjectionViewer(dome, 'Wagner VI', colors='base')
    assert len(np.unique(viewer.face_rgba, axis=0)) == n_parents
    viewer.set_view(-33.9, 151.2)
    viewer.set_colors(np.arange(len(dome.get_all_vertices()), dtype=float))      # per stored vertex
    plt.close(viewer.fig)
    for rotation in _random_rotations(3):
        assert _uncovered_pixels(SphereMap.from_manifold(dome, DEFAULT_PROJECTIONS['Equal Earth']), rotation) == 0
