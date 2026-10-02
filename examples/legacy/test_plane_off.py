# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.

from mt.geodesicdome.grid.plane import Lattice, Plane


def main():
    plane: Plane = Plane(10, 10, Lattice.Hexagonal)
    # plane: Plane = Plane(10, 10, Lattice.Rectilinear)
    v1 = plane.get_vertex_at(9, 0)
    neighbour = plane.get_neighbours(v1, False)
    if neighbour is not None:
        for v in neighbour:
            print(f'{v.x}, {v.y}')

        plane.unmark_vertices()
        lists = plane.get_neighbours_in_distance(v1, 2)
        print(f'lists.length = {len(lists)}')
        for vlist in lists:
            print(f'vlist size = {len(vlist)}')
            for i in range(len(vlist)):
                print(f'{vlist[i].x}, {vlist[i].y}')

        plane._write_off()


if __name__ == '__main__':
    main()
