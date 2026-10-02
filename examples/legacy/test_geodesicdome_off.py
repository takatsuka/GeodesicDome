# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.

from mt.geodesicdome.grid.geodesicdome import GeodesicDome, GeodesicVertex
from mt.geodesicdome.projection.kavrayskiy import KavrayskiyVII
from mt.geodesicdome.projection.projection import IProjection


def _write_OFF(dome: GeodesicDome):
    """
    writes the geodesicdome into a OFF format file.

    :return: None
    """
    allvertices: list[GeodesicVertex] = dome.get_all_vertices()
    triangles: list[GeodesicVertex] = dome.get_faces()
    ver_per_face = dome.get_number_of_vertices_per_face()

    with open('geodesicdome.off', 'w') as f:
        f.write('# OFF Data\n')
        f.write('OFF\n')
        f.write(f'{len(allvertices)} {int(len(triangles) / ver_per_face)} {len(allvertices)} \n')

        for i in range(len(allvertices)):
            v = allvertices[i]
            f.write(f'{v.coord[0]} {v.coord[1]} {v.coord[2]} \n')

        for i in range(len(triangles)):
            if i % ver_per_face == 0:
                f.write(f'{ver_per_face} ')
                for j in range(ver_per_face):
                    f.write(f'{triangles[i + j].id} ')
                f.write("\n")


def _write_plane_OFF(dome: GeodesicDome) -> None:
    """
    writes the geodesicdome into a OFF format file.  The created geometry is based on
    the rectilinear grid and you can see how vertices/triangles are generated.

    :return: None
    """
    allvertices: list[GeodesicVertex] = dome.get_all_vertices()
    triangles: list[GeodesicVertex] = dome.get_faces()
    ver_per_face = dome.get_number_of_vertices_per_face()

    with open('geodesicdome_plane.off', 'w') as f:
        f.write('# OFF Data\n')
        f.write('OFF\n')
        f.write(f'{len(allvertices)} {int(len(triangles) / ver_per_face)} {len(allvertices)} \n')

        for i in range(len(allvertices)):
            v = allvertices[i]
            f.write(f'{v.x} {v.y} {0} \n')

        for i in range(len(triangles)):
            if i % ver_per_face == 0:
                f.write(f'{ver_per_face} ')
                for j in range(ver_per_face):
                    f.write(f'{triangles[i + j].id} ')
                f.write("\n")


def _writeProjection_OFF(dome: GeodesicDome):
    allvertices: list[GeodesicVertex] = dome.get_all_vertices()
    # projection: IProjection = WagnerVI()
    projection: IProjection = KavrayskiyVII()
    triangles2D = projection.build(dome)

    with open('geodesicdome_WagnerVI.off', 'w') as f:
        f.write('# OFF Data\n')
        f.write('OFF\n')
        f.write(f'{len(allvertices)} {len(triangles2D)} {len(allvertices)} \n')

        for i in range(len(allvertices)):
            v = allvertices[i]
            twoD = projection.xyz_to_2d(v.coord)
            f.write(f'{twoD[0]} {twoD[1]} {0} \n')

        for v in triangles2D:
            f.write(f'3 {v[0]} {v[1]} {v[2]} \n')


def main():
    dome: GeodesicDome = GeodesicDome(1)
    dome.split(7)
    _write_OFF(dome)
    _write_plane_OFF(dome)
    _writeProjection_OFF(dome)

    dome._unmark_vertices()
    v1 = dome.get_vertex_at(3, 4)
    print(f'{v1.x}, {v1.y}')
    neighbour = dome.get_neighbours(v1, False)
    if neighbour:
        for v in neighbour:
            print(f'{v.x}, {v.y}')

    dome._unmark_vertices()
    lists = dome.get_neighbours_in_distance(v1, 2)
    print(f'lists.length = {len(lists)}')
    for vlist in lists:
        print(f'vlist size = {len(vlist)}')
        for i in range(len(vlist)):
            print(f'{vlist[i].x}, {vlist[i].y}')


if __name__ == '__main__':
    main()
