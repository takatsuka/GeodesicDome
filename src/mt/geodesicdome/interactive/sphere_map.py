# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
SphereMap: project a rotated spherical triangle mesh onto a 2D map, seam-correctly.

`Projection.build` simply drops the triangles that wrap around the +-180 degree meridian,
which leaves a ragged gap.  That is fine for one fixed picture but very visible while the
sphere is rotating, so SphereMap handles the awkward triangles properly:

* a triangle that straddles the +-180 degree meridian is drawn twice, once shifted by
  +-360 degrees of longitude, and both copies are clipped to the map outline;
* a triangle that contains a pole becomes a polygon that runs along the pole line;
* a triangle with a vertex exactly on a pole becomes a quadrilateral.

All four projections in mt.geodesicdome.projection are linear in longitude at a fixed
latitude, so the shifted copies line up exactly with the map edge.
"""

import numpy as np

from mt.geodesicdome.interactive.mesh import unique_mesh

TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi
_POLE_EPS = 1e-12
POLYGON_WIDTH = 6                                  # vertices per output polygon
_POLE_DIR = np.array([3.1e-7, 1.7e-7, 1.0])       # +z, nudged off any mesh edge


def _jitter_rotation(angle=1e-6, axis=(0.267, 0.534, 0.802)) -> np.ndarray:
    """A fixed, invisibly small rotation about a generic axis (Rodrigues' formula)."""
    k = np.asarray(axis) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


# Symmetric views (e.g. the un-rotated dome) put mesh edges exactly through the poles and
# vertices exactly on the +-180 degree meridian, where 'which way round' is undecidable.
# Rotating the mesh by 1e-6 rad (far below one pixel) makes every view generic.
_JITTER = _jitter_rotation()


def _wrap(angle):
    """Wraps angles into [-pi, pi)."""
    return (angle + np.pi) % TWO_PI - np.pi


class SphereMap:
    """
    A spherical triangle mesh plus a map projection.

    :param points: (V, 3) unit vectors
    :param faces: (F, 3) vertex indices, counter-clockwise seen from outside
    :param projection: a mt.geodesicdome.projection Projection instance
    :param max_edge: longest edge (radians) drawn as one straight segment.  A projected
                     edge is curved, so long edges are drawn through extra points: when the
                     mesh has longer edges, every face is drawn as n^2 smaller triangles
                     (display only; `faces` and the face indices are unchanged).  This keeps
                     coarse meshes, e.g. low-frequency or tetrahedral domes, free of gaps.
    """

    def __init__(self, points: np.ndarray, faces: np.ndarray, projection, max_edge: float = 0.2):
        self.points = np.asarray(points, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.projection = projection
        self.index_map = None           # stored-vertex id -> row in points (set by from_manifold)
        self.max_edge = float(max_edge)
        self._render = None             # (points, faces, parent face) actually drawn

    @classmethod
    def from_manifold(cls, manifold, projection, max_edge: float = 0.2) -> 'SphereMap':
        points, faces, index_map = unique_mesh(manifold)
        sphere_map = cls(points, faces, projection, max_edge=max_edge)
        sphere_map.index_map = index_map
        return sphere_map

    def _render_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The mesh that is drawn: the faces themselves, or each cut into n^2 triangles."""
        if self._render is not None:
            return self._render
        p, f = self.points, self.faces
        ends = p[f[:, [0, 1, 2]]], p[f[:, [1, 2, 0]]]
        longest = np.arccos(np.clip(np.einsum('fkj,fkj->fk', *ends), -1.0, 1.0)).max() if len(f) else 0.0
        n = max(1, int(np.ceil(longest / self.max_edge - 1e-9)))
        if n == 1:
            self._render = (p, f, np.arange(len(f)))
            return self._render
        # barycentric grid of one face; edge points are shared, so neighbouring faces meet exactly
        ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        local = {key: k for k, key in enumerate(ij)}
        w = np.array([[n - i - j, i, j] for i, j in ij], dtype=float) / n
        tris = []
        for i in range(n):
            for j in range(n - i):
                tris.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                if i + j <= n - 2:
                    tris.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))
        tris = np.array(tris)
        pts = np.einsum('gk,fkj->fgj', w, p[f])                    # (F, G, 3)
        pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
        pts, inverse = np.unique(np.round(pts.reshape(-1, 3), 12), axis=0, return_inverse=True)
        inverse = inverse.reshape(len(f), len(ij))
        faces = inverse[:, tris].reshape(-1, 3)
        parent = np.repeat(np.arange(len(f)), len(tris))
        self._render = (pts, faces, parent)
        return self._render

    # ------------------------------------------------------------------ basics
    def project(self, lat, lon) -> np.ndarray:
        """Projects lat/lon arrays (radians); returns shape lat.shape + (2,)."""
        return self.projection.latlong_to_2d(lat, lon)

    def outline(self, n: int = 181) -> np.ndarray:
        """The map boundary as an (2n, 2) polygon."""
        return self.projection.outline(n)

    def rotated_latlon(self, rotation: np.ndarray,
                       points: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rotated points, lat, lon) of every vertex (or of `points`) after applying `rotation`."""
        p = (self.points if points is None else points) @ (np.asarray(rotation) @ _JITTER).T
        lat = np.arcsin(np.clip(p[:, 2], -1.0, 1.0))
        lon = np.arctan2(p[:, 1], p[:, 0])
        return p, lat, lon

    # ---------------------------------------------------------------- polygons
    def polygons(self, rotation: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Projects every face of the mesh rotated by `rotation` (3x3, default identity).

        :return: (polygons, face_index)
                 polygons   -- (P, 6, 2) array in map coordinates.  Every polygon has
                               POLYGON_WIDTH = 6 vertices; triangles repeat their last
                               corner, which draws nothing extra.  Some polygons extend
                               beyond the map outline and must be clipped to it (see
                               `outline()`) -- that is how seam-crossing faces are drawn.
                 face_index -- (P,) index of the face each polygon came from
                               (a face can produce up to three polygons)
        """
        if rotation is None:
            rotation = np.eye(3)
        points, f, parent = self._render_mesh()
        p, lat, lon = self.rotated_latlon(rotation, points)
        at_pole = np.abs(p[:, 2]) > 1.0 - _POLE_EPS

        # Does the face contain the north/south pole?  For a counter-clockwise spherical
        # triangle (a, b, c), the pole n lies inside iff (a x b).n, (b x c).n, (c x a).n > 0.
        # The pole direction is nudged by ~1e-7 rad so that a pole lying exactly on an edge
        # (which happens, e.g. for the un-rotated dome) is owned by exactly one face.
        a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        d = np.stack([np.cross(a, b) @ _POLE_DIR, np.cross(b, c) @ _POLE_DIR,
                      np.cross(c, a) @ _POLE_DIR], axis=1)
        touches_pole = at_pole[f].any(axis=1)
        north = (d > 0).all(axis=1) & ~touches_pole
        south = (d < 0).all(axis=1) & ~touches_pole
        regular = ~(north | south | touches_pole)

        lat_rows: list[np.ndarray] = []           # (k, 6) latitude rows of the polygons
        lon_rows: list[np.ndarray] = []           # (k, 6) longitude rows
        face_index: list[np.ndarray] = []

        # --- ordinary faces (vectorised) -----------------------------------
        ids = np.nonzero(regular)[0]
        flat = lat[f[ids]]
        flon = lon[f[ids]]
        # Unwrap the corners.  Each edge takes the short way round (|step| < pi), except
        # that a face without a pole must wind zero times in total: if an edge passing
        # very close to a pole was wrapped the 'wrong' way, the steps sum to +-2pi, and
        # that edge (the one whose step is nearest +-pi) is corrected.
        steps = _wrap(np.roll(flon, -1, axis=1) - flon)           # a->b, b->c, c->a
        total = steps.sum(axis=1)
        worst = np.argmax(np.abs(steps), axis=1)
        steps[np.arange(len(steps)), worst] -= total
        flon = np.stack([flon[:, 0], flon[:, 0] + steps[:, 0], flon[:, 0] + steps[:, 0] + steps[:, 1]], axis=1)
        pad = [0, 1, 2, 2, 2, 2]                                  # triangle -> 6 vertices
        flat, flon = flat[:, pad], flon[:, pad]
        lat_rows.append(flat)
        lon_rows.append(flon)
        face_index.append(ids)
        for shift, mask in ((-TWO_PI, flon.max(axis=1) > np.pi), (TWO_PI, flon.min(axis=1) < -np.pi)):
            if mask.any():
                lat_rows.append(flat[mask])
                lon_rows.append(flon[mask] + shift)
                face_index.append(ids[mask])

        # --- the few faces at the poles ------------------------------------
        for fi in np.nonzero(~regular)[0]:
            corner_lat, corner_lon = lat[f[fi]], lon[f[fi]]
            if touches_pole[fi]:
                poly_lat, poly_lon = self._pole_vertex_polygon(corner_lat, corner_lon, at_pole[f[fi]])
            else:
                poly_lat, poly_lon = self._pole_face_polygon(corner_lat, corner_lon, HALF_PI if north[fi] else -HALF_PI)
            fill = np.r_[np.arange(len(poly_lat)), np.full(POLYGON_WIDTH - len(poly_lat), len(poly_lat) - 1)]
            poly_lat, poly_lon = poly_lat[fill], poly_lon[fill]
            for shift in (0.0, -TWO_PI, TWO_PI):
                shifted = poly_lon + shift
                if shifted.min() < np.pi and shifted.max() > -np.pi:
                    lat_rows.append(poly_lat[None])
                    lon_rows.append(shifted[None])
                    face_index.append(np.array([fi]))

        return self.project(np.vstack(lat_rows), np.vstack(lon_rows)), parent[np.concatenate(face_index)]

    @staticmethod
    def _pole_face_polygon(corner_lat, corner_lon, pole_lat):
        """A face that contains a pole: its boundary winds once around in longitude."""
        o = np.argsort(corner_lon)
        la, lo = corner_lat[o], corner_lon[o]
        poly_lat = np.array([la[0], la[1], la[2], la[0], pole_lat, pole_lat])
        poly_lon = np.array([lo[0], lo[1], lo[2], lo[0] + TWO_PI, lo[0] + TWO_PI, lo[0]])
        return poly_lat, poly_lon

    @staticmethod
    def _pole_vertex_polygon(corner_lat, corner_lon, corner_at_pole):
        """A face with one vertex on a pole: the pole vertex stretches along the pole line."""
        k = int(np.argmax(corner_at_pole))
        i, j = (k + 1) % 3, (k + 2) % 3
        lon_i = corner_lon[i]
        lon_j = lon_i + _wrap(corner_lon[j] - lon_i)
        pole_lat = np.sign(corner_lat[k]) * HALF_PI
        poly_lat = np.array([corner_lat[i], corner_lat[j], pole_lat, pole_lat])
        poly_lon = np.array([lon_i, lon_j, lon_j, lon_i])
        return poly_lat, poly_lon

    # ---------------------------------------------------------------- graticule
    def graticule(self, rotation: np.ndarray = None, step_deg: float = 30.0, samples: int = 181) -> np.ndarray:
        """
        Latitude/longitude lines of the ORIGINAL sphere, rotated and projected, as one
        (K, 2) array with NaN rows separating the pieces (ready for Line2D.set_data).
        Watching these lines bend is what makes the rotation easy to follow.
        """
        if rotation is None:
            rotation = np.eye(3)
        t = np.linspace(-np.pi, np.pi, samples)
        s = np.linspace(-HALF_PI, HALF_PI, samples)
        lines = []
        for lat0 in np.radians(np.arange(-90 + step_deg, 90, step_deg)):         # parallels
            lines.append(np.stack([np.cos(lat0) * np.cos(t), np.cos(lat0) * np.sin(t),
                                   np.full_like(t, np.sin(lat0))], 1))
        for lon0 in np.radians(np.arange(-180, 180, step_deg)):                  # meridians
            lines.append(np.stack([np.cos(s) * np.cos(lon0), np.cos(s) * np.sin(lon0), np.sin(s)], 1))

        pieces = []
        for xyz in lines:
            q = xyz @ np.asarray(rotation).T
            lat = np.arcsin(np.clip(q[:, 2], -1, 1))
            lon = np.arctan2(q[:, 1], q[:, 0])
            xy = self.project(lat, lon)
            jump = np.abs(np.diff(lon)) > np.pi                  # crossed the map edge
            xy = np.insert(xy, np.nonzero(jump)[0] + 1, np.nan, axis=0)
            pieces.extend([xy, np.full((1, 2), np.nan)])
        return np.vstack(pieces)

    # ------------------------------------------------------------------ helpers
    def face_centres(self) -> np.ndarray:
        """(F, 3) unit vectors at the centre of every face (original orientation)."""
        centres = self.points[self.faces].mean(axis=1)
        return centres / np.linalg.norm(centres, axis=1, keepdims=True)
