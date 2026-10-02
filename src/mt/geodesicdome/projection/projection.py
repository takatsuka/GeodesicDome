# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABCMeta, abstractmethod

import numpy
import numpy as np
from numpy import array, ndarray

from mt.geodesicdome import util


class IProjection:
    pass


class Projection(IProjection, metaclass=ABCMeta):
    @abstractmethod
    def _latlong_to_2d(self, latlon: array) -> array:
        raise NotImplementedError()

    def xyz_to_2d(self, coord: array) -> array:
        return self._latlong_to_2d(util.xyz_to_latlong(coord))

    def latlong_to_2d(self, lat, lon) -> ndarray:
        """
        Vectorised projection of latitude/longitude (radians, +z = north pole).

        :param lat: scalar or array of latitudes in [-pi/2, pi/2]
        :param lon: scalar or array of longitudes (same shape as lat).  Values outside
                    [-pi, pi] are extrapolated, which is handy for drawing shapes that
                    straddle the +-180 degree meridian.
        :return: array of shape lat.shape + (2,) holding the projected (x, y)
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        xy = self._latlong_to_2d(np.array([lat, lon]))
        return np.stack([xy[0], xy[1]], axis=-1)

    def xyz_to_2d_many(self, xyz: ndarray) -> ndarray:
        """
        Vectorised version of xyz_to_2d for an (N, 3) array of unit vectors.

        :return: (N, 2) array of projected coordinates
        """
        xyz = np.asarray(xyz, dtype=float)
        lat = np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0))
        lon = np.arctan2(xyz[..., 1], xyz[..., 0])
        return self.latlong_to_2d(lat, lon)

    def outline(self, n: int = 181) -> ndarray:
        """
        Returns the boundary of the projected world (the +-180 degree meridians,
        joined along the pole lines where the projection has them) as an (2n, 2) polygon.
        """
        lat = np.linspace(-np.pi / 2, np.pi / 2, n)
        right = self.latlong_to_2d(lat, np.full(n, np.pi))
        left = self.latlong_to_2d(lat[::-1], np.full(n, -np.pi))
        return np.vstack([right, left])

    def build(self, dome) -> ndarray:
        triangles = dome.get_faces()
        ver_per_face = dome.get_number_of_vertices_per_face()

        # project every vertex at once (vectorised), then keep the triangles facing the viewer
        vertices = dome.get_all_vertices()
        xy = self.xyz_to_2d_many(np.array([v.coord for v in vertices])).reshape(len(vertices), 2)
        for v, p in zip(vertices, xy, strict=True):
            v.projected_coord = p

        ids = np.array([v.id for v in triangles], dtype=int).reshape(-1, ver_per_face)[:, :3]
        row = np.empty(max(v.id for v in vertices) + 1 if vertices else 0, dtype=int)
        row[[v.id for v in vertices]] = np.arange(len(vertices))
        p1, p2, p3 = (xy[row[ids[:, k]]] for k in range(3))
        # z-component of (p2 - p1) x (p3 - p2) > 0  <=>  counter-clockwise (see util.facing)
        v1, v2 = p2 - p1, p3 - p2
        keep = (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) > 0
        return numpy.array(ids[keep].tolist())

