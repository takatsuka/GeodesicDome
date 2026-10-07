# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABCMeta

import numpy as np
from numpy import ndarray


class IManifold:
    pass

class Vertex(metaclass=ABCMeta):  # noqa: B024  (base class for GeodesicVertex etc.; no abstract methods)
    # Fixed attributes live in slots (smaller and faster than a per-instance dict); '__dict__'
    # keeps arbitrary extra attributes working, and the dict is only created when one is set.
    # (`coord` is not a slot: GeodesicVertex serves it from its dome's coordinate array.)
    __slots__ = ('visited', 'x', 'y', 'manifold', 'data', 'color', 'id', '__dict__', '__weakref__')

    def __init__(self, x: int, y: int):
        self.visited: bool = False
        self.x: int = x
        self.y: int = y
        self.manifold: IManifold = None
        self.data = None
        self.color = None
        self.id: int = -1
        self.coord: ndarray = np.array([0.0, 0.0, 0.0])

    def get_neighbours_in_distance(self, src_vertex, distance):
        if self.manifold is not None:
            return self.manifold.get_neighbours_in_distance(src_vertex, distance)
        return None

    def set_data(self, data):
        self.data = data
