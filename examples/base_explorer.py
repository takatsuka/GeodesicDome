# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
BaseExplorer: one figure that shows a geodesic dome three ways at once.

    3-D sphere   |   index-grid net   |   map projection centred on the chosen vertex

A centre vertex and its k neighbour rings are highlighted in all three panels, so you can
watch a neighbourhood that is compact on the sphere split across the seams of the net
(every stored copy of a ring vertex is marked).  Click a point on the net or on the map to
move the centre.  Used by examples/11_interactive_base_polyhedra.py and by the notebook
examples/notebooks/11_interactive_base_polyhedra.ipynb.

    from base_explorer import BaseExplorer
    ex = BaseExplorer(fig, base='tetrahedron', frequency=6, rings=3)
    ex.connect()                     # click to choose the centre
    ex.set_base('dodecahedron')      # set_frequency, set_rings, set_colouring, select(...)
"""
import dome_utils  # noqa: F401  (adds ../src to sys.path when run from a checkout)
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.grid.polyhedra import BASES, base_net, normalise_base
from mt.geodesicdome.interactive import SphereMap, unique_mesh
from mt.geodesicdome.interactive import rotation as rot
from mt.geodesicdome.interactive.viewer import DEFAULT_PROJECTIONS

COLOURINGS = ('base face', 'position', 'cell area')
SHEAR = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])      # index grid (x, y) -> plane
LIGHT = np.array([0.4, 0.5, 0.8]) / np.linalg.norm([0.4, 0.5, 0.8])
RING_CMAP = colormaps['plasma']


def spherical_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of every spherical triangle (unit sphere)."""
    a, b, c = (points[faces[:, k]] for k in range(3))
    triple = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    denom = 1 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    return 2 * np.arctan2(triple, denom)


class BaseExplorer:
    """
    :param fig: the matplotlib figure to draw in (its axes are created here)
    :param base: 'tetrahedron', 'icosahedron' or 'dodecahedron' (or 'tetra', 'icosa', 'dodeca')
    :param frequency: dome frequency
    :param rings: number of neighbour rings to highlight (0 = only the centre)
    :param colouring: one of COLOURINGS
    :param projection: a name in DEFAULT_PROJECTIONS
    :param rect: (left, bottom, width, height) of the figure area used by the three panels
    """

    def __init__(self, fig, base='icosahedron', frequency=4, rings=3, colouring='base face',
                 projection='Equal Earth', rect=(0.0, 0.0, 1.0, 1.0)):
        self.fig = fig
        self.base = normalise_base(base)
        self.frequency = int(frequency)
        self.rings = int(rings)
        self.colouring = colouring
        self.projection = projection
        self.rect = rect
        self.centre = None                    # a stored GeodesicVertex
        self._callbacks = []
        self._cid = None
        self._owned = []                      # axes and texts this explorer drew (other axes are left alone)
        self._build()
        self.select_kind('interior')

    # ------------------------------------------------------------------ public API
    def set_base(self, base):
        self.base = normalise_base(base)
        self._rebuild_keep_centre()

    def set_frequency(self, frequency):
        self.frequency = int(frequency)
        self._rebuild_keep_centre()

    def set_rings(self, rings):
        self.rings = int(rings)
        self.draw()

    def set_colouring(self, colouring):
        if colouring not in COLOURINGS:
            raise ValueError(f'colouring must be one of {COLOURINGS}')
        self.colouring = colouring
        self.draw()

    def set_projection(self, projection):
        self.projection = projection
        self.sphere_map = SphereMap.from_manifold(self.dome, DEFAULT_PROJECTIONS[projection])
        self.draw()

    def select(self, vertex):
        """Makes a stored vertex (or the nearest vertex to an (3,) direction) the centre."""
        if not hasattr(vertex, 'coord'):
            d = np.asarray(vertex, dtype=float)
            vertex = self.vertices[int(np.argmax(self.xyz @ (d / np.linalg.norm(d))))]
        self.centre = vertex
        self.draw()

    def select_kind(self, kind):
        """Centre on a base-solid 'corner', a 'seam' vertex (stored twice or more) or an 'interior' one."""
        degree = self.degree[self.index_map]
        seam = np.array([bool(v.same_vertices) for v in self.vertices])
        if kind == 'corner':
            candidates = np.nonzero(degree < 6)[0]
        elif kind == 'seam':
            candidates = np.nonzero(seam & (degree == 6))[0]
        else:
            candidates = np.nonzero(~seam & (degree == 6))[0]
        if len(candidates) == 0:                      # e.g. frequency 1 has no interior vertex
            candidates = np.arange(len(self.vertices))
        middle = self.grid.mean(axis=0)               # the candidate nearest the middle of the net
        best = candidates[np.argmin(np.linalg.norm(self.grid[candidates] - middle, axis=1))]
        self.select(self.vertices[best])

    def connect(self):
        """Click on the net or the map to choose the centre vertex."""
        if self._cid is None:
            self._cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def on_change(self, callback):
        """callback(explorer) after every redraw."""
        self._callbacks.append(callback)

    def summary(self) -> str:
        v = self.centre
        sizes = [len(r) for r in self.ring_lists]
        return (f"GeodesicDome({self.frequency}, base='{self.base}'): {len(self.points)} points, "
                f'{len(self.faces)} triangles, {len(self.vertices)} stored · centre ({v.x}, {v.y}) '
                f'with {self.degree[self.index_map[v.id]]} neighbours · ring sizes {sizes}')

    # ------------------------------------------------------------------ building
    def _build(self):
        self.dome = GeodesicDome(self.frequency, base=self.base)
        self.vertices = self.dome.get_all_vertices()
        self.stored_faces = self.dome.get_all_triangles().reshape(-1, 3)   # also sets vertex.id
        self.points, self.faces, self.index_map = unique_mesh(self.dome)
        self.xyz = self.dome.get_all_xyz()
        self.grid = np.array([[v.x, v.y] for v in self.vertices], dtype=float)
        self.net = self.grid @ SHEAR
        self.degree = np.zeros(len(self.points), dtype=int)
        edges = {tuple(sorted(e)) for f in self.faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
        for a, b in edges:
            self.degree[a] += 1
            self.degree[b] += 1
        centres = self.points[self.faces].mean(axis=1)
        self.face_centres = centres / np.linalg.norm(centres, axis=1, keepdims=True)
        self.parent = np.argmax(self.face_centres @ base_net(self.base).face_centres.T, axis=1)
        self.areas = spherical_areas(self.points, self.faces)
        self.sphere_map = SphereMap.from_manifold(self.dome, DEFAULT_PROJECTIONS[self.projection])

    def _rebuild_keep_centre(self):
        direction = None if self.centre is None else self.centre.coord.copy()
        self._build()
        if direction is None:
            self.select_kind('interior')
        else:
            self.select(direction)

    # ------------------------------------------------------------------ colours
    def _base_colours(self):
        if self.colouring == 'position':
            rgb = 0.35 + 0.6 * (0.5 + 0.5 * self.face_centres)
            return np.column_stack([rgb, np.ones(len(rgb))]), None
        if self.colouring == 'cell area':
            norm = Normalize(0.0, 2.0)
            return colormaps['RdBu_r'](norm(self.areas / self.areas.mean())), norm
        c = colormaps['tab20'](self.parent % 20)
        c[:, :3] = 0.45 + 0.55 * c[:, :3]                      # pastel, so the rings stand out
        return c, None

    def _rings(self):
        self.dome.unmark_vertices()
        self.ring_lists = self.dome.get_neighbours_in_distance(self.centre, self.rings) if self.rings else []
        self.dome.unmark_vertices()
        level = np.full(len(self.points), -1)
        level[self.index_map[self.centre.id]] = 0
        for k, ring in enumerate(self.ring_lists, start=1):
            level[self.index_map[[v.id for v in ring]]] = k
        return level

    # ------------------------------------------------------------------ drawing
    def draw(self):
        level = self._rings()
        face_rgba, area_norm = self._base_colours()
        corner_level = level[self.faces]
        inside = (corner_level >= 0).all(axis=1)
        top = max(self.rings, 1)
        ring_rgba = RING_CMAP(0.15 + 0.7 * corner_level.max(axis=1) / top)
        if self.colouring != 'cell area':
            face_rgba = np.where(inside[:, None], ring_rgba, face_rgba)
        point_level = level[self.index_map]                  # per stored vertex

        for artist in self._owned:
            artist.remove()
        left, bottom, width, height = self.rect
        w = width / 3
        ax3d = self.fig.add_axes([left, bottom + 0.06 * height, w, 0.84 * height], projection='3d')
        axnet = self.fig.add_axes([left + w + 0.01 * width, bottom + 0.10 * height, w - 0.02 * width, 0.76 * height])
        axmap = self.fig.add_axes([left + 2 * w + 0.01 * width, bottom + 0.10 * height, w - 0.02 * width,
                                   0.76 * height])
        self.ax3d, self.axnet, self.axmap = ax3d, axnet, axmap
        self._owned = [ax3d, axnet, axmap]

        # --- 3-D sphere (drag to rotate when the figure is live)
        tri = self.points[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        lit = 0.65 + 0.35 * np.clip(normals @ LIGHT, 0, 1)
        shaded = face_rgba.copy()
        shaded[:, :3] *= lit[:, None]
        ax3d.add_collection3d(Poly3DCollection(tri, facecolors=shaded, edgecolors=(1, 1, 1, 0.5),
                                               linewidths=0.3 if len(tri) < 4000 else 0.0))
        c = self.centre.coord * 1.02
        ax3d.scatter(*c, s=40, c='black', depthshade=False, zorder=10)
        ax3d.set_xlim(-1, 1), ax3d.set_ylim(-1, 1), ax3d.set_zlim(-1, 1)
        ax3d.set_box_aspect((1, 1, 1))
        elev = np.degrees(np.arcsin(np.clip(self.centre.coord[2], -1, 1)))
        azim = np.degrees(np.arctan2(self.centre.coord[1], self.centre.coord[0]))
        ax3d.view_init(elev=elev, azim=azim)                 # look at the centre vertex
        ax3d.set_axis_off()
        ax3d.set_title('on the sphere (drag to turn)', fontsize=10)

        # --- index-grid net: every stored copy of a ring vertex is marked
        # stored triangles are in the same order as the unique faces (unique_mesh keeps it)
        axnet.add_collection(PolyCollection(self.net[self.stored_faces], facecolors=face_rgba,
                                            edgecolors=(1, 1, 1, 0.6), linewidths=0.3))
        corner = self.degree[self.index_map] < 6
        axnet.scatter(*self.net[corner].T, s=34, c='#c0392b', marker='p', zorder=4)
        marked = point_level > 0
        axnet.scatter(*self.net[marked].T, s=10, c=RING_CMAP(0.15 + 0.7 * point_level[marked] / top),
                      edgecolors='black', linewidths=0.3, zorder=5)
        copies = point_level == 0
        axnet.scatter(*self.net[copies].T, s=46, c='black', marker='*', zorder=6)
        axnet.set_aspect('equal')
        axnet.autoscale()
        axnet.set_axis_off()
        axnet.set_title(f'index grid {self.dome.x_max + 1} x {self.dome.y_max + 1} (click to choose the centre)',
                        fontsize=10)

        # --- map projection centred on the vertex
        lat, lon = self._latlon(self.centre.coord)
        self._rotation = rot.view_rotation(lat, lon)
        polygons, face_index = self.sphere_map.polygons(self._rotation)
        outline = self.sphere_map.outline()
        clip = Polygon(outline, closed=True, facecolor='none', edgecolor='#333333', linewidth=1.0, zorder=3)
        axmap.add_patch(clip)
        faces = PolyCollection(polygons, facecolors=face_rgba[face_index], edgecolors=face_rgba[face_index],
                               linewidths=0.3)
        faces.set_clip_path(clip)
        axmap.add_collection(faces)
        g = self.sphere_map.graticule(self._rotation)
        (line,) = axmap.plot(g[:, 0], g[:, 1], color='black', lw=0.5, alpha=0.35)
        line.set_clip_path(clip)
        axmap.plot(0, 0, marker='*', ms=10, color='black')
        axmap.set_aspect('equal')
        axmap.set_xlim(outline[:, 0].min() * 1.02, outline[:, 0].max() * 1.02)
        axmap.set_ylim(outline[:, 1].min() * 1.05, outline[:, 1].max() * 1.05)
        axmap.set_axis_off()
        axmap.set_title(f'{self.projection}, centred on the vertex (click to move)', fontsize=10)

        if area_norm is not None:
            cax = self.fig.add_axes([left + 2 * w + 0.25 * w, bottom + 0.08 * height, 0.5 * w, 0.02 * height])
            self._owned.append(cax)
            cb = self.fig.colorbar(ScalarMappable(norm=area_norm, cmap='RdBu_r'), cax=cax, orientation='horizontal')
            cb.set_label('cell area / mean cell area', fontsize=8)
            cb.ax.tick_params(labelsize=7)
        self._owned.append(self.fig.text(left + width / 2, bottom + 0.955 * height, self.summary(),
                                         ha='center', fontsize=10))
        self.fig.canvas.draw_idle()
        for callback in self._callbacks:
            callback(self)

    # ------------------------------------------------------------------ events
    @staticmethod
    def _latlon(p):
        return float(np.arcsin(np.clip(p[2], -1, 1))), float(np.arctan2(p[1], p[0]))

    def _on_click(self, event):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if event.xdata is None or (toolbar is not None and getattr(toolbar, 'mode', '')):
            return
        click = np.array([event.xdata, event.ydata])
        if event.inaxes is self.axnet:
            self.select(self.vertices[int(np.argmin(np.linalg.norm(self.net - click, axis=1)))])
        elif event.inaxes is self.axmap:
            p, lat, lon = self.sphere_map.rotated_latlon(self._rotation)
            xy = self.sphere_map.project(lat, lon)
            self.select(self.points[int(np.argmin(np.linalg.norm(xy - click, axis=1)))])


__all__ = ['BaseExplorer', 'BASES', 'COLOURINGS', 'spherical_areas']
