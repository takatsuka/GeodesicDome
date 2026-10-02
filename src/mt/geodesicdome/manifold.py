# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

from mt.geodesicdome.vertex import Vertex


class IManifold:
    pass


class Manifold(IManifold, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_all_vertices(self) -> list[Vertex]:
        raise NotImplementedError()

    @abstractmethod
    def get_faces(self) -> list[Vertex]:
        raise NotImplementedError()

    @abstractmethod
    def get_number_of_vertices_per_face(self) -> int:
        raise NotImplementedError()

    def get_neighbours_in_distance(self, v: Vertex, dis: int) -> list[list[Vertex]]:
        """
        Returns 2D array containing neighbours at different levels.
        :param v:
        :param dis:
        :return:
        """
        n1: list[Vertex] = self.get_neighbours(v, False)
        neighbours: list[list[Vertex]] = [n1]
        level: int = dis - 1
        for i in range(level):
            n1: list[Vertex] = []
            for vTmp in neighbours[i]:
                n = self.get_neighbours(vTmp, False)
                n1.extend(n)
            neighbours.append(n1)

        return neighbours

    @abstractmethod
    def get_vertex_at(self, x, y) -> Vertex:
        """
        Returns a vertex at (x, y) rectilinear location.

        :param x: x-coordinate
        :param y: y-coordinate
        :return: Vertex found at (x, y)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_neighbours(self, vertex: Vertex, visit_same_vertex: bool = False) -> list[Vertex]:
        """
        This function will find and return the immediate neighbours of the specified vertex.

        :param vertex: The vertex whose immediate neighbours will be found.
        :param visit_same_vertex: If this is False (default), registered same vertices are not checked.
        :return: A list of the immediate neighbours of the vertex.
        """
        raise NotImplementedError()

    def unmark_vertices(self) -> None:
        """
        Sets all visited flag to False

        :return: None
        """
        for pv in self.get_all_vertices():
            pv.visited = False

    def get_all_xyz(self) -> ndarray:
        xyz = []
        for v in self.get_all_vertices():
            xyz.append(v.coord)
        return np.array(xyz)

    def get_all_triangles(self) -> ndarray:
        xyz = []
        for v in self.get_faces():
            xyz.append(v.id)
        return np.array(xyz)
