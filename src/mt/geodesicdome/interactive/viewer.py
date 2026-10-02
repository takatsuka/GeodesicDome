# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
ProjectionViewer: an interactive map of a geodesic dome that you rotate with the mouse.

    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    from mt.geodesicdome.interactive import ProjectionViewer

    ProjectionViewer(GeodesicDome(8)).show()

Mouse and keyboard
    drag (left button)   rotate the sphere: left/right spins it about the map's polar
                         axis, up/down tilts it towards/away from you (a fast preview is
                         drawn while dragging; full quality returns on release)
    arrow keys           rotate by 5 degrees (hold shift for 1 degree)
    , / .                roll the view about the map centre
    r                    reset the view
    p                    next projection (the radio buttons on the left do the same)
    g / e                toggle the graticule / the triangle edges

Needs a GUI matplotlib backend: a plain `python script.py` on a desktop works, and in
Jupyter use `%matplotlib widget` (pip install ipympl).
"""
from collections.abc import Callable, Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba_array
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import Button, RadioButtons

from mt.geodesicdome.interactive import rotation as rot
from mt.geodesicdome.interactive.sphere_map import SphereMap
from mt.geodesicdome.projection.equal_earth import EqualEarth
from mt.geodesicdome.projection.kavrayskiy import KavrayskiyVII
from mt.geodesicdome.projection.projection import Projection
from mt.geodesicdome.projection.wagner import WagnerIII, WagnerVI

DEFAULT_PROJECTIONS: dict[str, Projection] = {
    'Equal Earth': EqualEarth(),
    'Kavrayskiy VII': KavrayskiyVII(),
    'Wagner VI': WagnerVI(),
    'Wagner III': WagnerIII(),
}

ColourSpec = Union[None, str, np.ndarray, Sequence]  # noqa: UP007  (runtime alias; keep typing.Union)


class ProjectionViewer:
    """
    Interactive, rotatable map projection of a GeodesicDome.

    :param dome: a GeodesicDome (any triangular Manifold on the unit sphere works)
    :param projection: a Projection instance or one of the names in DEFAULT_PROJECTIONS
                       (default 'Equal Earth')
    :param colors: how to colour the faces
        * 'position' (default) -- a fixed colour from each face's original xyz position,
          so you can see where every part of the sphere has moved
        * 'base'               -- colour by the parent face of the dome's own base solid
                                  (4, 20 or 12 colours for a tetrahedron, icosahedron or
                                  dodecahedron dome)
        * 'icosahedron'        -- colour by the parent icosahedron face (20 colours)
        * 'data'               -- use vertex.data of every vertex (RGB/RGBA or scalars)
        * a single colour      -- e.g. 'lightsteelblue'
        * an array             -- per stored vertex (len N), per unique point (len 10f²+2)
                                  or per face (len 20f²); scalars go through `cmap`,
                                  (…, 3|4) arrays are used as RGB(A)
    :param cmap, vmin, vmax: colour map and limits for scalar values
    :param edges: draw triangle edges (default: on for frequency <= 12)
    :param graticule: draw the original sphere's latitude/longitude lines (default on)
    :param view: initial (lat, lon) in degrees of the original point at the map centre
    :param projections: dict name -> Projection offered by the radio buttons
    :param title: figure title
    :param figsize: matplotlib figure size
    """

    def __init__(self, dome, projection: str | Projection | None = None, *,
                 colors: ColourSpec = 'position', cmap: str = 'viridis',
                 vmin: float | None = None, vmax: float | None = None,
                 edges: bool | None = None, graticule: bool = True,
                 view: Sequence[float] | None = None,
                 projections: dict[str, Projection] | None = None,
                 title: str | None = None, figsize=(12, 6.5)):
        self.dome = dome
        self.projections = dict(projections or DEFAULT_PROJECTIONS)
        self.projection_name, projection = self._resolve_projection(projection)
        self.map = SphereMap.from_manifold(dome, projection)

        self.cmap = plt.get_cmap(cmap)
        self.norm: Normalize | None = None
        self.face_rgba = self._resolve_colours(colors, vmin, vmax)

        frequency = getattr(dome, 'frequency', 1)
        self.show_edges = frequency <= 12 if edges is None else bool(edges)
        self.show_graticule = bool(graticule)

        self.rotation = np.eye(3) if view is None else rot.view_rotation(*np.radians(view))
        self._home = self.rotation.copy()
        self._drag_from = None
        self._background = None               # cached figure without the map, for blitting
        self._vertex_buffer = None             # (capacity, 7, 2) coordinates shared by ...
        self._path_pool: list[Path] = []       # ... these reusable Path objects
        self._callbacks: list[Callable[[ProjectionViewer], None]] = []

        self._build_figure(title, figsize)
        self._redraw_outline()
        self.update()

    # ================================================================ public API
    def show(self):
        """Opens the window (blocks until it is closed when not in interactive mode)."""
        plt.show()

    def save(self, path: str, **kwargs):
        """Saves the current view to an image file."""
        self.fig.savefig(path, **kwargs)

    def rotate(self, d_lon: float = 0.0, d_lat: float = 0.0, roll: float = 0.0):
        """
        Rotates the sphere, in degrees: `d_lon` about the map's polar axis (like a
        horizontal drag), `d_lat` about the east-west axis through the map centre (like a
        vertical drag), `roll` about the map centre.
        """
        r = rot.rot_x(np.radians(roll)) @ rot.drag_rotation(np.radians(d_lon), np.radians(d_lat))
        self.set_rotation(r @ self.rotation)

    def set_view(self, lat: float, lon: float, roll: float = 0.0):
        """Puts the original-sphere point (lat, lon) [degrees] at the map centre."""
        self.set_rotation(rot.view_rotation(*np.radians([lat, lon, roll])))

    def set_rotation(self, matrix: np.ndarray):
        """Sets the 3x3 rotation (original sphere -> view) directly."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3) or np.linalg.det(matrix) <= 0:
            raise ValueError('rotation must be a 3x3 rotation matrix (determinant +1, not a mirror image)')
        self.rotation = rot.orthonormalise(matrix)
        self.update()

    def reset(self):
        """Back to the initial view."""
        self.set_rotation(self._home)

    def set_projection(self, projection: str | Projection):
        """Switches the map projection (an instance or a name from `projections`)."""
        self.projection_name, self.map.projection = self._resolve_projection(projection)
        if self.projection_name in self.projections and self._radio is not None:
            labels = [t.get_text() for t in self._radio.labels]
            index = labels.index(self.projection_name)
            if self._radio.value_selected != self.projection_name:
                self._radio.set_active(index)       # triggers _on_radio -> comes back here
                return
        self._redraw_outline()
        self.update()

    def set_colors(self, colors: ColourSpec, vmin: float | None = None, vmax: float | None = None):
        """Changes the face colours (same options as the constructor's `colors`)."""
        self.face_rgba = self._resolve_colours(colors, vmin, vmax)
        self.update()

    def on_rotate(self, callback: Callable[['ProjectionViewer'], None]):
        """Registers callback(viewer), called after every change of the view."""
        self._callbacks.append(callback)

    @property
    def centre(self):
        """(lat, lon) in degrees of the original-sphere point at the map centre."""
        return tuple(float(a) for a in np.degrees(rot.centre_of_view(self.rotation)))

    def update(self):
        """Re-projects the rotated sphere and redraws."""
        polygons, face_index = self.map.polygons(self.rotation)
        self._set_polygons(polygons)
        colours = self.face_rgba[face_index]
        self._faces.set_facecolor(colours)
        if self._drag_from is not None:
            # fast preview while dragging: plain aliased fill (no strokes, no seams)
            self._faces.set_antialiased(False)
            self._faces.set_edgecolor('none')
            self._faces.set_linewidth(0)
        else:
            # full quality: antialiased, with edges or with face-coloured hairlines that
            # hide the seams antialiasing would otherwise leave between triangles
            self._faces.set_antialiased(True)
            self._faces.set_edgecolor((1, 1, 1, 0.55) if self.show_edges else colours)
            self._faces.set_linewidth(0.4 if self.show_edges else 0.3)

        if self.show_graticule:
            g = self.map.graticule(self.rotation)
            self._graticule.set_data(g[:, 0], g[:, 1])
        self._graticule.set_visible(self.show_graticule)

        if self._background is not None:
            # dragging: repaint only the map over the cached background (blitting)
            canvas = self.fig.canvas
            canvas.restore_region(self._background)
            for artist in (self._faces, self._graticule, self._outline):
                self.ax.draw_artist(artist)
            canvas.blit(self.ax.bbox)
        else:
            lat, lon = self.centre
            self._info.set_text(f'{self.projection_name}   centre of view: lat {lat:+6.1f}°, lon {lon:+7.1f}°'
                                '     drag to rotate · arrows · r reset · p projection · g grid · e edges')
            self.fig.canvas.draw_idle()
        for callback in self._callbacks:
            callback(self)

    # ============================================================ figure set-up
    def _build_figure(self, title, figsize):
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.manager and self.fig.canvas.manager.set_window_title('mt.geodesicdome ProjectionViewer')
        has_colorbar = self.norm is not None
        self.ax = self.fig.add_axes([0.16, 0.08, 0.80 if not has_colorbar else 0.74, 0.84])
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()

        self._outline = Polygon(np.zeros((3, 2)), closed=True, facecolor='none',
                                edgecolor='#333333', linewidth=1.2, zorder=3)
        self.ax.add_patch(self._outline)
        self._faces = PolyCollection([], antialiased=True, zorder=1)
        self._faces.set_clip_path(self._outline)
        self.ax.add_collection(self._faces)
        (self._graticule,) = self.ax.plot([], [], color='black', lw=0.6, alpha=0.45, zorder=2)
        self._graticule.set_clip_path(self._outline)
        self._info = self.fig.text(0.5, 0.025, '', ha='center', fontsize=9, color='#444444')
        if title:
            self.fig.suptitle(title, fontsize=13)

        # projection chooser and reset button
        self._radio = None
        names = list(self.projections)
        if names:
            rax = self.fig.add_axes([0.01, 0.55, 0.13, 0.05 * len(names) + 0.03], frameon=False)
            active = names.index(self.projection_name) if self.projection_name in names else 0
            self._radio = RadioButtons(rax, names, active=active)
            self._radio.on_clicked(self._on_radio)
        bax = self.fig.add_axes([0.02, 0.45, 0.10, 0.05])
        self._reset_button = Button(bax, 'Reset view')
        self._reset_button.on_clicked(lambda _event: self.reset())

        if has_colorbar:
            cax = self.fig.add_axes([0.92, 0.2, 0.015, 0.6])
            self.fig.colorbar(plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap), cax=cax)

        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('key_press_event', self._on_key)

    def _redraw_outline(self):
        outline = self.map.outline()
        self._outline.set_xy(outline)
        pad = 0.03 * (outline[:, 0].max() - outline[:, 0].min())
        self.ax.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
        self.ax.set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
        self._map_size = (np.ptp(outline[:, 0]), np.ptp(outline[:, 1]))

    # ========================================================== event handlers
    def _pixels_per_radian(self):
        """How many screen pixels one radian of longitude / latitude spans on the map."""
        (x0, y0), (x1, y1) = self.ax.transData.transform([[0.0, 0.0], list(self._map_size)])
        return abs(x1 - x0) / (2 * np.pi), abs(y1 - y0) / np.pi

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self._drag_from = (event.x, event.y)
            self._background = self._grab_background()

    def _on_motion(self, event):
        if self._drag_from is None or event.x is None:
            return
        px_lon, px_lat = self._pixels_per_radian()
        dx, dy = event.x - self._drag_from[0], event.y - self._drag_from[1]
        self._drag_from = (event.x, event.y)
        self.set_rotation(rot.drag_rotation(dx / px_lon, dy / px_lat) @ self.rotation)

    def _on_release(self, event):
        if self._drag_from is not None:
            self._drag_from = None
            self._background = None
            self.update()                      # redraw at full quality

    def _set_polygons(self, polygons: np.ndarray):
        """
        Hands the (P, 6, 2) polygons to the PolyCollection.  Rebuilding thousands of
        Path objects every frame is the slowest part of a redraw, so a pool of Paths
        is kept whose vertices are views into one buffer, and only the numbers change.
        """
        n, width = polygons.shape[0], polygons.shape[1] + 1
        if self._vertex_buffer is None or len(self._vertex_buffer) < n:
            capacity = int(n * 1.25) + 16
            self._vertex_buffer = np.zeros((capacity, width, 2))
            codes = np.full(width, Path.LINETO, dtype=Path.code_type)
            codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
            self._path_pool = [Path(self._vertex_buffer[i], codes) for i in range(capacity)]
            if not np.shares_memory(self._path_pool[0].vertices, self._vertex_buffer):
                self._vertex_buffer = None          # this matplotlib copies: use the plain API
        if self._vertex_buffer is None:
            self._faces.set_verts(polygons)
            return
        self._vertex_buffer[:n, :-1] = polygons
        self._vertex_buffer[:n, -1] = polygons[:, 0]
        self._faces._paths = self._path_pool[:n]
        self._faces.stale = True

    def _grab_background(self):
        """Renders the figure without the map artists and keeps a copy of the map area."""
        canvas = self.fig.canvas
        if not getattr(canvas, 'supports_blit', False):
            return None
        artists = (self._faces, self._graticule, self._outline)
        visible = [a.get_visible() for a in artists]
        for a in artists:
            a.set_visible(False)
        try:
            canvas.draw()
            background = canvas.copy_from_bbox(self.ax.bbox)
        except Exception:                      # backend without blitting support
            background = None
        finally:
            for a, v in zip(artists, visible, strict=False):
                a.set_visible(v)
        return background

    def _on_key(self, event):
        key = event.key or ''
        step = 1.0 if key.startswith('shift+') else 5.0
        key = key.replace('shift+', '')
        moves = {'left': (-step, 0, 0), 'right': (step, 0, 0), 'up': (0, step, 0),
                 'down': (0, -step, 0), ',': (0, 0, -step), '.': (0, 0, step)}
        if key in moves:
            self.rotate(*moves[key])
        elif key == 'r':
            self.reset()
        elif key == 'p' and self.projections:
            names = list(self.projections)
            current = names.index(self.projection_name) if self.projection_name in names else -1
            self.set_projection(names[(current + 1) % len(names)])
        elif key == 'g':
            self.show_graticule = not self.show_graticule
            self.update()
        elif key == 'e':
            self.show_edges = not self.show_edges
            self.update()

    def _on_radio(self, label):
        self.projection_name, self.map.projection = self._resolve_projection(label)
        self._redraw_outline()
        self.update()

    # ================================================================= helpers
    def _resolve_projection(self, projection):
        if projection is None:
            name = next(iter(self.projections), None)
            return (name, self.projections[name]) if name else ('Equal Earth', EqualEarth())
        if isinstance(projection, str):
            if projection not in self.projections:
                raise ValueError(f'unknown projection {projection!r}; choose from {list(self.projections)}')
            return projection, self.projections[projection]
        for name, p in self.projections.items():
            if p is projection or type(p) is type(projection):
                return name, projection
        return type(projection).__name__, projection

    def _resolve_colours(self, colors: ColourSpec, vmin, vmax) -> np.ndarray:
        """Returns an (F, 4) RGBA array, one row per face of self.map."""
        m = self.map
        n_faces, n_points = len(m.faces), len(m.points)
        n_stored = len(m.index_map) if m.index_map is not None else -1
        self.norm = None

        if colors is None or (isinstance(colors, str) and colors == 'position'):
            c = m.face_centres()
            rgb = 0.5 + 0.5 * c                         # xyz in [-1, 1] -> rgb in [0, 1]
            rgb = 0.35 + 0.6 * rgb                      # soften
            return np.column_stack([rgb, np.ones(n_faces)])
        if isinstance(colors, str) and colors == 'icosahedron':
            from mt.geodesicdome.grid.geodesicdome import GeodesicDome
            ico = SphereMap.from_manifold(GeodesicDome(1), m.projection)
            parent = np.argmax(m.face_centres() @ ico.face_centres().T, axis=1)
            return plt.cm.tab20(parent % 20)
        if isinstance(colors, str) and colors == 'base':
            from mt.geodesicdome.grid.polyhedra import base_net
            centres = base_net(getattr(self.dome, 'base', 'icosahedron')).face_centres
            parent = np.argmax(m.face_centres() @ centres.T, axis=1)
            return plt.cm.tab20(parent % 20)
        if isinstance(colors, str) and colors == 'data':
            values = [v.data for v in self.dome.get_all_vertices()]
            if any(v is None for v in values):
                raise ValueError("colors='data' needs vertex.set_data(...) on every vertex")
            colors = np.array(values, dtype=float)
        if isinstance(colors, str) or (np.ndim(colors) == 1 and len(colors) in (3, 4)
                                       and n_faces not in (3, 4) and n_points not in (3, 4)):
            return np.tile(to_rgba_array(colors)[0], (n_faces, 1))

        values = np.asarray(colors, dtype=float)
        if len(values) == n_faces:
            per_face = values
        else:
            if len(values) == n_stored:
                per_point = np.zeros((n_points,) + values.shape[1:])
                per_point[m.index_map] = values
            elif len(values) == n_points:
                per_point = values
            else:
                raise ValueError(f'colour array has {len(values)} rows; expected {n_stored} (stored vertices), '
                                 f'{n_points} (unique points) or {n_faces} (faces)')
            per_face = per_point[m.faces].mean(axis=1)

        if per_face.ndim == 1:                           # scalars -> colour map
            self.norm = Normalize(per_face.min() if vmin is None else vmin,
                                  per_face.max() if vmax is None else vmax)
            return self.cmap(self.norm(per_face))
        if per_face.shape[1] == 3:
            per_face = np.column_stack([per_face, np.ones(len(per_face))])
        return np.clip(per_face, 0, 1)
