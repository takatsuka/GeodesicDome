# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.

import numpy as np
from numpy import array

from mt.geodesicdome import util
from mt.geodesicdome.manifold import Manifold
from mt.geodesicdome.vertex import Vertex

from .polyhedra import BASES, BaseNet, base_net, normalise_base


class IGeodesicDome:
    pass


class GeodesicVertex(Vertex):
    def __init__(self, latitude=None, longitude=None, coord=None, x=None, y=None, frequency=1):
        super().__init__(x, y)
        self.vertices: list[list[GeodesicVertex]] = [[]]
        self.same_vertices: list[GeodesicVertex] = None
        self.triangles: list[GeodesicVertex] = []
        self.frequency = frequency
        self.coord: array
        self.projected_coord: array
        self.latlon_coord: array
        if (latitude is not None) and (longitude is not None):
            self.coord = util.spherical_to_xyz(latitude, longitude)
            self.latlon_coord = np.array([latitude, longitude])
        elif coord.any():
            self.coord = coord
            self.latlon_coord = util.xyz_to_spherical(coord)


def _mark_same_vertices(v: GeodesicVertex, visit_or_not: bool) -> None:
    """
    This function sets the same visited status to all 'same' vertices.

    :type v: GeodesicVertex
    :param v: The vertex whose registered same vertices will have the specified visited status.
    :type visit_or_not: bool
    :param visit_or_not: the new visited or not status.
    :return: None
    """
    if v.same_vertices is not None and v.same_vertices:
        for vSame in v.same_vertices:
            vSame.visited = visit_or_not


def _partition(frequency: int, v1: GeodesicVertex, v2: GeodesicVertex) -> list[GeodesicVertex]:
    """
    This function will generate (frequency - 1) new vertices between v1 and v2 so that
    the edge (v1, v2) are equally divided.

    :param frequency: The frequency of the Geodesicdome.
           For example, frequency 2 will divide an edge into two edges by inserting one vertex.
    :param v1: One of two vertices defining an edge.
    :param v2: One of two vertices defining an edge.
    :return: A list of newly created (incerted) vertices.
    """
    dx = (v2.coord[0] - v1.coord[0]) / frequency
    dy = (v2.coord[1] - v1.coord[1]) / frequency
    dz = (v2.coord[2] - v1.coord[2]) / frequency
    new_vertices: list[GeodesicVertex] = [None] * (frequency - 1)  # GeodesicVertex[frequency - 1];
    dxIndex = int((v2.x - v1.x) / frequency)
    dyIndex = int((v2.y - v1.y) / frequency)
    # print(f'(dxIndex, dyIndex): {dxIndex}, {dyIndex}')
    for j in range(1, frequency):
        new_coord = np.array([v1.coord[0] + j * dx, v1.coord[1] + j * dy, v1.coord[2] + j * dz])
        new_coord = new_coord / np.linalg.norm(new_coord)
        x = v1.x + dxIndex * j
        y = v1.y + dyIndex * j
        # print(f'(x, y): {x}, {y}')
        new_vertices[j - 1] = GeodesicVertex(coord=new_coord, x=x, y=y, frequency=frequency)

    return new_vertices


def _split_x_vertices(vertices_x: list[GeodesicVertex], frequency: int) -> list[GeodesicVertex]:
    # new vector containing the new higher frequency vertex
    vNum = len(vertices_x) + (frequency - 1) * (len(vertices_x) - 1)
    new_vertices_x: list[GeodesicVertex] = [None] * vNum
    v1: GeodesicVertex = vertices_x[0]

    # these will shift x and y coordinates to make room for new vertices.
    v1.x *= frequency
    v1.y *= frequency
    offset = v1.y

    for j in range(1, len(vertices_x)):
        # create the higher frequency vertex
        v2: GeodesicVertex = vertices_x[j]
        v2.x *= frequency
        v2.y *= frequency

        created: list[GeodesicVertex] = _partition(frequency, v1, v2)
        new_vertices_x[v1.y - offset] = v1
        for k in range(len(created)):
            new_vertices_x[v1.y + k + 1 - offset] = created[k]
        new_vertices_x[v1.y + frequency - offset] = v2
        v1 = v2

    return new_vertices_x


class GeodesicDome(IGeodesicDome, Manifold):
    """
    A geodesic dome: a base polyhedron whose faces are cut into an f-frequency triangular
    grid and pushed out onto the unit sphere.

        GeodesicDome(8)                          # icosahedron (default): 10 f^2 + 2 points
        GeodesicDome(8, base='tetrahedron')      # 4HSOM tetrahedron:      2 f^2 + 2 points
        GeodesicDome(8, base='dodecahedron')     # pentakis dodecahedron: 30 f^2 + 2 points

    ``base`` accepts 'tetrahedron' / 'tetra', 'icosahedron' / 'icosa', 'dodecahedron' /
    'dodeca' (see :func:`mt.geodesicdome.grid.polyhedra.normalise_base`).  Calling
    ``GeodesicDome(...)`` returns an :class:`IcosahedronDome`, :class:`TetrahedronDome` or
    :class:`DodecahedronDome`; all of them are GeodesicDomes with the same interface:
    vertices on an integer index grid (``vertex.x``, ``vertex.y``) whose six neighbours are
    the offsets (+-1, 0), (0, +-1), (+1, +1), (-1, -1), seam copies in
    ``vertex.same_vertices``, ``get_vertex_at``, ``get_neighbours``,
    ``get_neighbours_in_distance``, ``get_faces``, ``split``.
    """

    base: str = 'icosahedron'

    def __new__(cls, frequency=1, base=None, *args, **kwargs):
        if cls is GeodesicDome:
            cls = _DOME_CLASSES[normalise_base(base)]
        return super().__new__(cls)

    def _check_base(self, base) -> None:
        if base is not None and normalise_base(base) != self.base:
            raise ValueError(f'{type(self).__name__} is built on the {self.base}, not on {base!r}')

    def get_all_vertices(self) -> list[GeodesicVertex]:
        all_vertices: list[GeodesicVertex] = []
        for row in self.vertices:
            all_vertices.extend(row)
        return all_vertices

    def get_number_of_vertices_per_face(self) -> int:
        return 3

    def get_faces(self) -> list[GeodesicVertex]:
        self._build_faces()
        return self.triangles

    def _updateIDs(self) -> None:
        serial_number = 0
        for v in self.get_all_vertices():
            v.id = serial_number
            serial_number += 1

    def _unmark_vertices(self) -> None:
        for x_list in self.vertices:
            for v in x_list:
                v.visited = False


class IcosahedronDome(GeodesicDome):
    """
    A Geodesicdome based on the Icosahedron (22 vertices and 20 triangles)
    The base Icosahedron's vertices are arranged on a rectilinear grid as shown below.
    This is to use an indexing scheme to quickly search vertex's i-level neighbours.

    #                  v17  v20  v22
    #             v13  v16  v19  v21
    #         v9  v12  v15  v18
    #     v5  v8  v11  v14
    # v2  v4  v7  v10
    # v1  v3  v6
    """

    """
    Icosahedron's approximated average arc length
    """
    _arc_length = 1.106588

    base = 'icosahedron'

    def __init__(self, frequency=1, base=None):
        super().__init__()
        self._check_base(base)
        self.arcLength = IcosahedronDome._arc_length  # approximate the average arc length
        # start from the base icosahedron (frequency 1); split() below multiplies this
        self.frequency = 1
        self.x_max = 6
        self.y_max = 5
        self.vertices: list[list[GeodesicVertex]] = [[]] * (self.x_max + 1)

        # initialize the icosahedron, calculate all the vertex coordinates
        dAzimuth = 2 * np.pi / 5
        dLatitude = ((90 - 26.565) / 180) * np.pi

        # set 22 vertices
        v1 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, 0), x=0, y=0, frequency=1)
        v2 = GeodesicVertex(coord=util.spherical_to_xyz(0, 0), x=0, y=1, frequency=1)
        vList1 = [v1, v2]
        self.vertices[0] = vList1

        v3 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 0.5), x=1, y=0, frequency=1)
        v4 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, dAzimuth), x=1, y=1, frequency=1)
        v5 = GeodesicVertex(coord=util.spherical_to_xyz(0, 0), x=1, y=2, frequency=1)
        vList2 = [v3, v4, v5]
        self.vertices[1] = vList2

        v6 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi, 0), x=2, y=0, frequency=1)
        v7 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 1.5), x=2, y=1, frequency=1)
        v8 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, dAzimuth * 2), x=2, y=2, frequency=1)
        v9 = GeodesicVertex(coord=util.spherical_to_xyz(0, 0), x=2, y=3, frequency=1)
        vList3 = [v6, v7, v8, v9]
        self.vertices[2] = vList3

        v10 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi, 0), x=3, y=1, frequency=1)
        v11 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 2.5), x=3, y=2, frequency=1)
        v12 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, dAzimuth * 3), x=3, y=3, frequency=1)
        v13 = GeodesicVertex(coord=util.spherical_to_xyz(0, 0), x=3, y=4, frequency=1)
        vList4 = [v10, v11, v12, v13]
        self.vertices[3] = vList4

        v14 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi, 0), x=4, y=2, frequency=1)
        v15 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 3.5), x=4, y=3, frequency=1)
        v16 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, dAzimuth * 4), x=4, y=4, frequency=1)
        v17 = GeodesicVertex(coord=util.spherical_to_xyz(0, 0), x=4, y=5, frequency=1)
        vList5 = [v14, v15, v16, v17]
        self.vertices[4] = vList5

        v18 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi, 0), x=5, y=3, frequency=1)
        v19 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 4.5), x=5, y=4, frequency=1)
        v20 = GeodesicVertex(coord=util.spherical_to_xyz(dLatitude, 0), x=5, y=5, frequency=1)
        vList6 = [v18, v19, v20]
        self.vertices[5] = vList6

        v21 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi, 0), x=6, y=4, frequency=1)
        v22 = GeodesicVertex(coord=util.spherical_to_xyz(np.pi - dLatitude, dAzimuth * 0.5), x=6, y=5, frequency=1)
        vList7 = [v21, v22]
        self.vertices[6] = vList7

        # initialize same vertices list for vertices
        # v1
        v1.same_vertices = [v20]

        # v2
        v2.same_vertices = [v5, v9, v13, v17]

        # v3
        v3.same_vertices = [v22]

        # v5
        v5.same_vertices = [v2, v9, v13, v17]

        # v6
        v6.same_vertices = [v10, v14, v18, v21]

        # v9
        v9.same_vertices = [v2, v5, v13, v17]

        # v10
        v10.same_vertices = [v6, v14, v18, v21]

        # v13
        v13.same_vertices = [v2, v5, v9, v17]

        # v14
        v14.same_vertices = [v6, v10, v18, v21]

        # v17
        v17.same_vertices = [v2, v5, v9, v13]

        # v18
        v18.same_vertices = [v6, v10, v14, v21]

        # v20
        v20.same_vertices = [v1]

        # v21
        v21.same_vertices = [v6, v10, v14, v18]

        # v22
        v22.same_vertices = [v3]

        if frequency > 1:
            self.split(frequency)

    def __repr__(self) -> str:
        return f'GeodesicDome(frequency={self.frequency})'

    def _find_same_vertices(self):
        """
            used after split, find the same vertices for the new generated vertices
                               *
                             / |
                            / /*
                           / / |
                   ^     *--*--*
                   |     |
                   |     *
                         |
        """
        first_x: list[GeodesicVertex] = self.vertices[0]
        top_current: GeodesicVertex = first_x[len(first_x) - 1]
        top_next: GeodesicVertex = None
        # stage1, travel through the top vertices that has 4 same vertices points
        # through v2, v5, v9, v13 and v17
        i = 0
        while i < len(top_current.same_vertices):
            top_next = top_current.same_vertices[i]
            x1 = top_current.x
            y1 = top_current.y
            x2 = top_next.x
            y2 = top_next.y
            top_middle: GeodesicVertex = self.get_vertex_at(x1 + 1, y1)
            while (x1 + 1) != x2:
                if top_middle.same_vertices is not None and top_middle.same_vertices:
                    x1 += 1
                    y2 -= 1
                    top_middle = self.get_vertex_at(x1 + 1, y1)
                    continue

                top_middle_match = self.get_vertex_at(x2, y2 - 1)
                top_middle.same_vertices = [top_middle_match]
                top_middle_match.same_vertices = [top_middle]

                x1 += 1
                y2 -= 1
                top_middle = self.get_vertex_at(x1 + 1, y1)

            top_current = top_next
            i += 1

        # stage 2, travel through the flat top vertex
        # through v20, v21
        v17_x = top_next.x  # v17
        v17_y = top_next.y
        v2 = self.vertices[0][len(self.vertices[0]) - 1]
        for i in range(v17_x + 1, self.x_max):
            vHigh = self.get_vertex_at(i, v17_y)
            if vHigh.same_vertices is not None and vHigh.same_vertices:
                continue

            xdiff = vHigh.x - v17_x
            vLow = self.get_vertex_at(0, v2.y - xdiff) if (xdiff < v2.y) else self.get_vertex_at(xdiff - v2.y, 0)

            vHigh.same_vertices = [vLow]
            vLow.same_vertices = [vHigh]

        # through v21 to v22
        last_x = self.vertices[self.x_max]
        v22 = last_x[len(last_x) - 1]
        v3 = v22.same_vertices[0]
        for i in range(len(last_x) - 1 - 1, 0, -1):
            test = last_x[i]
            if test.same_vertices is not None and test.same_vertices:
                continue
            match = self.get_vertex_at(v3.x + (v22.y - test.y), 0)
            match.same_vertices = [test]
            test.same_vertices = [match]

        # stage 3, travel through the boottom vertice that has 4 same vertices
        # through v21, v18, v14, v10, v6
        v21 = last_x[0]
        v1 = v21
        i = len(v1.same_vertices) - 1
        while i >= 0:
            v2 = v1.same_vertices[i]  # v18
            x1 = v1.x
            y1 = v1.y
            x2 = v2.x
            y2 = v2.y
            v3: GeodesicVertex = self.get_vertex_at(x1 - 1, y1)
            while (x1 - 1) != x2:
                # print(f'x1, x2, y1, y2, v2, v3 = {x1, x2, y1, y2, v2, v3}')
                if v3.same_vertices is not None and v3.same_vertices:
                    x1 -= 1
                    y2 += 1
                    v3 = self.get_vertex_at(x1 - 1, y1)
                    continue

                v4 = self.get_vertex_at(x2, y2 + 1)
                v3.same_vertices = [v4]
                v4.same_vertices = [v3]

                x1 -= 1
                y2 += 1
                v3 = self.get_vertex_at(x1 - 1, y1)

            v1 = v2
            i -= 1

    # increase frequency
    def split(self, frequency):
        """
        Subdivides every edge of the current dome into `frequency` segments.
        Calls are cumulative: split(2) followed by split(3) gives frequency 6.

        :param frequency: subdivision factor (an int >= 2)
        :return: None
        """
        self.frequency *= frequency
        self.x_max *= frequency
        self.y_max *= frequency
        self.arcLength /= frequency

        # temporary new vector for storing the vertices of the new frequency
        new_vertices: list[list[GeodesicVertex]] = [[]] * (self.x_max + 1)

        # split the first vector of x-vertices
        new_x_vertices1: list[GeodesicVertex] = _split_x_vertices(self.vertices[0], frequency)
        new_vertices[0] = new_x_vertices1
        # print(f'# of columns  : {len(self.vertices)}')
        for i in range(len(self.vertices) - 1):
            # print(f'processing index : {i}')
            new_x_vertices2: list[GeodesicVertex] = _split_x_vertices(self.vertices[i + 1], frequency)
            new_vertices[(i + 1) * frequency] = new_x_vertices2

            # create the vertices between the 2 new vectors
            RB = new_x_vertices2[0]  # right bottom
            LT = new_x_vertices1[len(new_x_vertices1) - 1]  # left top
            length = (LT.y - RB.y + 1)

            # prepare (x-vertices * (frequency -1)) to be inserted
            for k in range(frequency - 1):
                index: int = i * frequency + k + 1
                new_vertices[index] = [None] * length

            # fill-in key horizontals.
            horiindex = []
            for n in range(len(new_x_vertices1) - 1, -1, -frequency):
                left: GeodesicVertex = new_x_vertices1[n]
                # print(f'left.y: {left.y}')
                right: GeodesicVertex = None
                right_y: int = 0
                for m in range(len(new_x_vertices2) - 1, -1, -1):
                    testing: GeodesicVertex = new_x_vertices2[m]
                    # print(f'testing: {testing.y}')
                    if testing.y == left.y:
                        right = testing
                        right_y = m
                        horiindex.append([n, m])
                        break
                if right is not None:
                    inserting: list[GeodesicVertex] = _partition(frequency, left, right)
                    for v in inserting:
                        new_vertices[v.x][right_y] = v

            # fill-in diagonals
            base_index = i * frequency
            for hi in range(len(horiindex) - 1):
                [h_index, m] = horiindex[hi]
                for h_count in range(2, frequency + 1):
                    b1 = new_vertices[base_index][h_index - h_count]
                    t1h = len(new_vertices[base_index + h_count]) - 1 - hi * frequency if h_count < frequency else m
                    t1 = new_vertices[base_index + h_count][t1h]
                    new_points1 = _partition(h_count, b1, t1)
                    for ii in range(len(new_points1)):
                        new_v = new_points1[ii]
                        new_vertices[new_v.x][t1h - h_count + 1 + ii] = new_v
                    if h_count < frequency:
                        b2 = new_vertices[base_index + frequency - h_count][t1h - frequency]
                        t2 = new_vertices[base_index + frequency][m - frequency + h_count]
                        new_points2 = _partition(h_count, b2, t2)
                        for jj in range(len(new_points2)):
                            new_v = new_points2[jj]
                            new_vertices[new_v.x][t1h - frequency + 1 + jj] = new_v

            # ready for the next iteration
            new_x_vertices1 = new_x_vertices2

        self.vertices = new_vertices
        self._find_same_vertices()

    def get_vertex_at(self, x, y) -> GeodesicVertex:
        if (x <= self.x_max) and (y <= self.y_max) and (x >= 0) and (y >= 0):
            xVector: list[GeodesicVertex] = self.vertices[x]

            # offset
            offset = xVector[0].y
            if y >= offset and y < (len(xVector) + offset):
                return xVector[y - offset]

    def _build_faces(self) -> list[GeodesicVertex]:
        self._updateIDs()
        vNum = 3 * 20 * self.frequency * self.frequency
        self.triangles: list[GeodesicVertex] = [None] * vNum
        index = 0
        for i in range(len(self.vertices) - 1):
            # print(f'builsing face at : {i}')
            current_x: list[GeodesicVertex] = self.vertices[i]
            next_x: list[GeodesicVertex] = self.vertices[i + 1]
            next_bottom_y = next_x[0].y
            for j in range(len(current_x) - 1):
                if current_x[j].y == next_bottom_y:  # found the starting point
                    for k in range(j, len(current_x) - 1):
                        #  triangle 1
                        self.triangles[index] = current_x[k]  # v1
                        self.triangles[index + 1] = next_x[k - j]  # v2
                        self.triangles[index + 2] = next_x[k - j + 1]  # v3

                        # triangle 2
                        self.triangles[index + 3] = self.triangles[index]
                        self.triangles[index + 4] = self.triangles[index + 2]
                        self.triangles[index + 5] = current_x[k + 1]  # v4
                        index += 6
                    break

        return self.triangles

    def get_neighbours(self, v: GeodesicVertex, visit_same_vertex: bool) -> list[GeodesicVertex]:
        v.visited = True
        x = v.x
        y = v.y
        if v.same_vertices:
            _mark_same_vertices(v, True)

        # find the neighbors
        neighbours: list[GeodesicVertex] = []
        xArray1 = self.vertices[v.x]
        offset1 = xArray1[0].y

        # x, y + 1
        if (y + 1) < (len(xArray1) + offset1):
            n1 = xArray1[y + 1 - offset1]
            if not n1.visited:
                n1.visited = True
                neighbours.append(n1)
                if n1.same_vertices:
                    _mark_same_vertices(n1, True)
        # x, y - 1
        if (y - 1) >= offset1:
            n4 = xArray1[y - 1 - offset1]
            if not n4.visited:
                n4.visited = True
                neighbours.append(n4)
                if n4.same_vertices:
                    _mark_same_vertices(n4, True)

        if (x + 1) <= self.x_max:
            xArray2 = self.vertices[v.x + 1]
            offset2 = xArray2[0].y

            # x+1, y
            if y >= offset2 and y < (len(xArray2) + offset2):
                n2 = xArray2[y - offset2]
                if not n2.visited:
                    n2.visited = True
                    neighbours.append(n2)
                    if n2.same_vertices:
                        _mark_same_vertices(n2, True)

            '''
            x+1, y+1 is most probably v's same vertex point
            if n3's is in v.sameVertics, markSameVertex(n3,frequency) would mark all
            the v's same vertices, so that the program would not searching for v's same Vertices's
            neighbor later
            '''
            if (y + 1) >= offset2 and (y + 1) < len(xArray2) + offset2:
                n3 = xArray2[y + 1 - offset2]
                if not n3.visited:
                    n3.visited = True
                    neighbours.append(n3)
                    if n3.same_vertices:
                        _mark_same_vertices(n3, True)
        if (x - 1) >= 0:
            xArray3 = self.vertices[v.x - 1]
            offset3 = xArray3[0].y
            '''
            x - 1, y - 1 is most probably v's same vertex point
            if n5's is in v.sameVerticsList, markSameVertex(n5,frequency) would mark all
            the v's same vertices, so that the program would not searching for v's same Vertices's
            neighbor later
            '''
            if (y - 1) < (len(xArray3) + offset3) and (y - 1) >= offset3:
                n5 = xArray3[y - 1 - offset3]
                if not n5.visited:
                    n5.visited = True
                    neighbours.append(n5)
                    if n5.same_vertices:
                        _mark_same_vertices(n5, True)

            # x - 1, y
            if y < (len(xArray3) + offset3) and y >= offset3:
                n6: GeodesicVertex = xArray3[y - offset3]
                if not n6.visited:
                    n6.visited = True
                    neighbours.append(n6)
                    if n6.same_vertices:
                        _mark_same_vertices(n6, True)

        # find the neighbor of the same vertex;
        if v.same_vertices is None or visit_same_vertex:
            return neighbours

        for vSame in v.same_vertices:
            vSameNeighbor = self.get_neighbours(vSame, True)
            neighbours.extend(vSameNeighbor)

        return neighbours


# grid offsets of the six neighbours, in the order IcosahedronDome.get_neighbours visits them
_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (1, 1), (-1, -1), (-1, 0))


class NetDome(GeodesicDome):
    """
    A geodesic dome laid out on the index grid by a :class:`~mt.geodesicdome.grid.polyhedra.BaseNet`.

    Every base triangle of the net is cut into an f-frequency grid; each grid point is
    interpolated on the flat face and projected onto the unit sphere.  A point that the net
    shows more than once (on a cut) is stored once per position, and the copies list each
    other in ``same_vertices``, exactly as in the icosahedral dome.

    Neighbour search is pure index arithmetic: a vertex at (x, y) looks at the six offsets
    (+-1, 0), (0, +-1), (+1, +1), (-1, -1) in the grid, keeping those whose edge lies inside
    the net (a 6-bit mask per vertex, set at construction), and then continues from its seam
    copies.
    """

    base: str = ''

    def __init__(self, frequency=1, base=None):
        super().__init__()
        self._check_base(base)
        self.net: BaseNet = base_net(self.base)
        self.frequency = 1
        self._build(1)
        if frequency > 1:
            self.split(frequency)

    def __repr__(self) -> str:
        return f'GeodesicDome(frequency={self.frequency}, base={self.base!r})'

    # ------------------------------------------------------------------ construction
    def split(self, frequency):
        """
        Subdivides every edge of the current dome into `frequency` segments.
        Calls are cumulative: split(2) followed by split(3) gives frequency 6.
        The vertices are rebuilt, so vertex objects from before the split are not reused.

        :param frequency: subdivision factor (an int >= 1)
        :return: None
        """
        frequency = int(frequency)
        if frequency < 1:
            raise ValueError('frequency must be a positive integer')
        if frequency > 1:
            self._build(self.frequency * frequency)

    def _build(self, f: int) -> None:
        net = self.net
        self.frequency = f
        self.x_max, self.y_max = net.grid_size(f)
        grid: list[list[GeodesicVertex | None]] = [[None] * (self.y_max + 1) for _ in range(self.x_max + 1)]
        keys: dict[tuple[int, int], tuple] = {}
        copies: dict[tuple, list[GeodesicVertex]] = {}
        triangles: list[tuple[GeodesicVertex, GeodesicVertex, GeodesicVertex]] = []

        for ids, corners, (o, a1, a2) in zip(net.triangles, net.points[net.triangles], net.lattice * f, strict=True):
            e1 = (a1 - o) // f                    # one grid step along each side of the triangle
            e2 = (a2 - o) // f
            local: dict[tuple[int, int], GeodesicVertex] = {}
            for i in range(f + 1):
                for j in range(f + 1 - i):
                    counts = (f - i - j, i, j)
                    key = tuple(sorted((int(ids[k]), counts[k]) for k in range(3) if counts[k]))
                    x, y = (int(c) for c in o + i * e1 + j * e2)
                    v = grid[x][y]
                    if v is None:
                        coord = (counts[0] * corners[0] + counts[1] * corners[1] + counts[2] * corners[2]) / f
                        v = GeodesicVertex(coord=coord / np.linalg.norm(coord), x=x, y=y, frequency=f)
                        grid[x][y] = v
                        keys[x, y] = key
                        copies.setdefault(key, []).append(v)
                    elif keys[x, y] != key:
                        raise AssertionError(f'{self.base} net overlaps itself at grid ({x}, {y})')
                    local[i, j] = v
            for i in range(f):
                for j in range(f - i):
                    triangles.append((local[i, j], local[i + 1, j], local[i, j + 1]))
                    if i + j <= f - 2:
                        triangles.append((local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]))

        # seam copies
        for group in copies.values():
            for v in group:
                v.same_vertices = [w for w in group if w is not v] if len(group) > 1 else None

        # which of the six grid offsets are edges of the net
        edges = set()
        for tri in triangles:
            for k in range(3):
                p, q = tri[k], tri[(k + 1) % 3]
                edges.add((p.x, p.y, q.x - p.x, q.y - p.y))
                edges.add((q.x, q.y, p.x - q.x, p.y - q.y))
        for column in grid:
            for v in column:
                if v is not None:
                    v.neighbour_mask = sum(1 << k for k, (dx, dy) in enumerate(_DIRECTIONS)
                                           if (v.x, v.y, dx, dy) in edges)

        self._grid = grid
        self.vertices = [[v for v in column if v is not None] for column in grid]
        self._net_triangles = triangles
        ends = np.array([[[p.coord, q.coord] for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))]
                         for t in triangles]).reshape(-1, 2, 3)
        cos = np.clip(np.einsum('ij,ij->i', ends[:, 0], ends[:, 1]), -1.0, 1.0)
        self.arcLength = float(np.arccos(cos).mean())   # mean great-circle edge length (radians)

    # ------------------------------------------------------------------ queries
    def get_vertex_at(self, x, y) -> GeodesicVertex:
        if 0 <= x <= self.x_max and 0 <= y <= self.y_max:
            return self._grid[x][y]
        return None

    def _build_faces(self) -> list[GeodesicVertex]:
        self._updateIDs()
        self.triangles = [v for tri in self._net_triangles for v in tri]
        return self.triangles

    def get_neighbours(self, v: GeodesicVertex, visit_same_vertex: bool = False) -> list[GeodesicVertex]:
        v.visited = True
        if v.same_vertices:
            _mark_same_vertices(v, True)

        neighbours: list[GeodesicVertex] = []
        grid, mask = self._grid, v.neighbour_mask
        for k, (dx, dy) in enumerate(_DIRECTIONS):
            if mask >> k & 1:
                n = grid[v.x + dx][v.y + dy]
                if not n.visited:
                    n.visited = True
                    neighbours.append(n)
                    if n.same_vertices:
                        _mark_same_vertices(n, True)

        if v.same_vertices is None or visit_same_vertex:
            return neighbours
        for v_same in v.same_vertices:
            neighbours.extend(self.get_neighbours(v_same, True))
        return neighbours


class TetrahedronDome(NetDome):
    """
    Geodesic dome on the tetrahedron, stored as the 4HSOM orthogonal array of
    R. M. de Sousa and R. C. L. Oliveira (IJCNN 2012): f + 1 rows (y = 0..f) by
    2f + 1 columns (x = 0..2f), every cell used.

        row f:   C . . . D . . . C       C at (0, f) and (2f, f); D (north pole) at (f, f)
        row 0:   A . . . B . . . A       A at (0, 0) and (2f, 0); B at (f, 0)

    Seams: column 0 = column 2f, and (i, 0) = (2f - i, 0), (i, f) = (2f - i, f).
    2f^2 + 2 unique points, 4f^2 triangles; the four corners A, B, C, D have 3 neighbours.
    """

    base = 'tetrahedron'


class DodecahedronDome(NetDome):
    """
    Geodesic dome on the dodecahedron: each pentagon is cut into 5 triangles at its centre
    (the pentakis dodecahedron) and each of those into an f-frequency grid, on the flat
    pentagon.  30f^2 + 2 unique points, 60f^2 triangles; the 12 pentagon centres have 5
    neighbours, every other point 6.  Frequency 1 is the pentakis dodecahedron (32 points).
    """

    base = 'dodecahedron'


_DOME_CLASSES = {'icosahedron': IcosahedronDome, 'tetrahedron': TetrahedronDome, 'dodecahedron': DodecahedronDome}
assert set(_DOME_CLASSES) == set(BASES)
