# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from enum import Enum

from mt.geodesicdome.manifold import Manifold
from mt.geodesicdome.vertex import Vertex


class Lattice(Enum):
    Hexagonal = 0
    Rectilinear = 1


class Topology(Enum):
    Plane = 0
    Donut = 1


class PlaneVertex(Vertex):

    def __init__(self, x: int, y: int, lattice=Lattice.Hexagonal):
        super().__init__(x, y)
        match lattice:
            case Lattice.Hexagonal:
                self.coord[1] = y
                self.coord[2] = 0
                if y % 2 == 0:
                    self.coord[0] = x
                else:
                    self.coord[0] = (x + 0.5)
            case Lattice.Rectilinear:
                self.coord[0] = x
                self.coord[1] = y
                self.coord[2] = 0


class Plane(Manifold):
    def __init__(self, x, y, lattice=Lattice.Hexagonal, topology=Topology.Plane):
        super().__init__()
        self.lattice: Lattice = lattice
        self.topology: Topology = topology
        self.x = x
        self.y = y
        self.vertices: list[Vertex] = [None] * self.x * self.y
        for i in range(y):
            for j in range(x):
                self.vertices[i * self.x + j] = PlaneVertex(j, i, self.lattice)
        self.faces: list[Vertex] = []

    def get_all_vertices(self) -> list[Vertex]:
        return self.vertices

    def get_vertex_at(self, x, y):
        if (x < self.x) and (y < self.y) and (x >= 0) and (y >= 0):
            index = y * self.x + x
            return self.vertices[index]
        return None

    def get_number_of_vertices_per_face(self) -> int:
        return 3 if self.lattice == Lattice.Hexagonal else 4

    def get_faces(self) -> list[Vertex]:
        # print(f'lattice: {self.lattice}')
        if self.lattice == Lattice.Hexagonal:
            self._build_faces_hex()
        elif self.lattice == Lattice.Rectilinear:
            self._build_faces_recti()
        return self.faces

    def get_neighbours(self, v: Vertex, visit_same_vertex: bool) -> list[Vertex]:
        v.visited = True
        x = v.x
        y = v.y

        # find the neighbors
        neighbours: list[Vertex] = []

        # x + 1, y
        x1 = x + 1
        if x1 >= self.x:
            if self.topology == Topology.Donut:
                x1 = 0
            elif self.topology == Topology.Plane:
                x1 = -1
        if x1 > -1:
            nv = self.vertices[y * self.x + x1]
            if not nv.visited:
                nv.visited = True
                neighbours.append(nv)

        # x - 1, y
        x_1 = x - 1
        if x_1 < 0:
            if self.topology == Topology.Donut:
                x_1 = self.x - 1
            elif self.topology == Topology.Plane:
                x_1 = -1
        if x_1 > -1:
            nv = self.vertices[y * self.x + x_1]
            if not nv.visited:
                nv.visited = True
                neighbours.append(nv)

        # x, y - 1
        y_1 = y - 1
        if y_1 < 0:
            if self.topology == Topology.Donut:
                y_1 = self.y - 1
            else:
                y_1 = -1
        if self.lattice == Lattice.Hexagonal:
            if y % 2 == 0:
                x1 = x - 1
                x2 = x
                if x1 < 0:
                    if self.topology == Topology.Donut:
                        x1 = self.x - 1
                    elif self.topology == Topology.Plane:
                        x1 = -1
                if x1 > -1 and y_1 > -1:
                    nv = self.vertices[y_1 * self.x + x1]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
                if x2 > -1 and y_1 > -1:
                    nv = self.vertices[y_1 * self.x + x2]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
            else:
                x1 = x
                x2 = x + 1
                if x2 > self.x - 1:
                    if self.topology == Topology.Donut:
                        x2 = 0
                    elif self.topology == Topology.Plane:
                        x2 = -1
                if x2 > -1 and y_1 > -1:
                    nv = self.vertices[y_1 * self.x + x2]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
                if x1 > -1 and y_1 > -1:
                    nv = self.vertices[y_1 * self.x + x1]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
        elif self.lattice == Lattice.Rectilinear:
            if y_1 > -1:
                nv = self.vertices[y_1 * self.x + x]
                if not nv.visited:
                    nv.visited = True
                    neighbours.append(nv)

        # x, y + 1
        y1 = y + 1
        if y1 > self.y - 1:
            if self.topology == Topology.Donut:
                y1 = 0
            else:
                y1 = -1
        if self.lattice == Lattice.Hexagonal:
            if y % 2 == 0:
                x1 = x - 1
                x2 = x
                if x1 < 0:
                    if self.topology == Topology.Donut:
                        x1 = self.x - 1
                    elif self.topology == Topology.Plane:
                        x1 = -1
                if x1 > -1 and y1 > -1:
                    nv = self.vertices[y1 * self.x + x1]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
                if x2 > -1 and y1 > -1:
                    nv = self.vertices[y1 * self.x + x2]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
            else:
                x1 = x
                x2 = x + 1
                if x2 > self.x - 1:
                    if self.topology == Topology.Donut:
                        x2 = 0
                    elif self.topology == Topology.Plane:
                        x2 = -1
                if x2 > -1 and y1 > -1:
                    nv = self.vertices[y1 * self.x + x2]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
                if x1 > -1 and y1 > -1:
                    nv = self.vertices[y1 * self.x + x1]
                    if not nv.visited:
                        nv.visited = True
                        neighbours.append(nv)
        elif self.lattice == Lattice.Rectilinear:
            if y1 > -1:
                nv = self.vertices[y1 * self.x + x]
                if not nv.visited:
                    nv.visited = True
                    neighbours.append(nv)

        return neighbours

    def _update_ids(self) -> None:
        serial_number = 0
        for v in self.vertices:
            v.id = serial_number
            serial_number += 1

    def _build_faces_hex(self) -> list[Vertex]:
        self._update_ids()
        vNum = (self.x - 1) * 2 * (self.y - 1) * 3
        self.faces: list[Vertex] = [None] * vNum
        index = 0
        for i in range(self.y - 1):
            for j in range(self.x - 1):
                self.faces[index] = self.vertices[i * self.x + j]
                if i % 2 == 0:
                    self.faces[index + 1] = self.vertices[i * self.x + (j + 1)]
                    self.faces[index + 2] = self.vertices[(i + 1) * self.x + j]

                    self.faces[index + 3] = self.vertices[i * self.x + (j + 1)]
                    self.faces[index + 4] = self.vertices[(i + 1) * self.x + (j + 1)]
                    self.faces[index + 5] = self.vertices[(i + 1) * self.x + j]
                else:
                    self.faces[index + 1] = self.vertices[i * self.x + (j + 1)]
                    self.faces[index + 2] = self.vertices[(i + 1) * self.x + (j + 1)]

                    self.faces[index + 3] = self.vertices[i * self.x + j]
                    self.faces[index + 4] = self.vertices[(i + 1) * self.x + (j + 1)]
                    self.faces[index + 5] = self.vertices[(i + 1) * self.x + j]

                index += 6
        return self.faces

    def _build_faces_recti(self) -> list[Vertex]:
        self._update_ids()
        vNum = (self.x - 1) * (self.y - 1)
        self.faces: list[Vertex] = [None] * vNum * self.get_number_of_vertices_per_face()
        index = 0
        for i in range(self.y - 1):
            for j in range(self.x - 1):
                self.faces[index] = self.vertices[i * self.x + j]
                self.faces[index + 1] = self.vertices[i * self.x + (j + 1)]
                self.faces[index + 2] = self.vertices[(i + 1) * self.x + (j + 1)]
                self.faces[index + 3] = self.vertices[(i + 1) * self.x + j]
                index += 4

        return self.faces

    def _write_off(self):
        allvertices: list[Vertex] = self.get_all_vertices()
        faces: list[Vertex] = self.get_faces()
        ver_per_face = self.get_number_of_vertices_per_face()

        with open('plane.off', 'w') as f:
            f.write('# OFF Data\n')
            f.write('OFF\n')
            f.write(f'{len(allvertices)} {int(len(faces) / ver_per_face)} {len(allvertices)} \n')

            for i in range(len(allvertices)):
                v = allvertices[i]
                f.write(f'{v.coord[0]} {v.coord[1]} {v.coord[2]} \n')

            for i in range(len(faces)):
                if i % ver_per_face == 0:
                    f.write(f'{ver_per_face} ')
                    for j in range(ver_per_face):
                        f.write(f'{faces[i + j].id} ')
                    f.write("\n")


