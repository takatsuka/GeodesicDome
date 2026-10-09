# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Checks that two GeodesicDome implementations give identical domes.

    git archive 6f664da src | tar -x -C /tmp/gd-old           # the pre-rewrite source
    python benchmarks/check_equivalence.py dump /tmp/old.pkl --src /tmp/gd-old/src
    python benchmarks/check_equivalence.py dump /tmp/new.pkl                     # this checkout
    python benchmarks/check_equivalence.py compare /tmp/old.pkl /tmp/new.pkl

44 cases: icosahedron, tetrahedron, dodecahedron and the generic engine on the icosahedron
net, at f = 1, 2, 3, 4, 5, 7, 8, 13, 16 and the cumulative splits 2*3 and 3*2*2.  Compared:
coordinates (bit for bit), triangles, grid positions, seam lists, latitude/longitude, faces,
column sizes, every vertex's neighbours (with and without seam copies, in order) and rings
up to distance 4 around ~40 vertices per dome.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1] / 'src'


def plain(cls, *args, **kwargs):
    """The unrelaxed dome (relax=False; older versions without the option are always unrelaxed)."""
    try:
        return cls(*args, relax=False, **kwargs)
    except TypeError:
        return cls(*args, **kwargs)


def dump(path, src):
    sys.path.insert(0, str(src or REPO_SRC))
    from mt.geodesicdome.grid.geodesicdome import GeodesicDome, NetDome

    class GenericIcosahedron(NetDome):
        base = 'icosahedron'

    cases = []
    for base in ('icosahedron', 'tetrahedron', 'dodecahedron', 'generic-icosahedron'):
        cases += [(base, (f,)) for f in (1, 2, 3, 4, 5, 7, 8, 13, 16)]
        cases += [(base, (2, 3)), (base, (3, 2, 2))]
    out = {}
    for base, seq in cases:
        cls = GenericIcosahedron if base == 'generic-icosahedron' else GeodesicDome
        d = plain(cls, seq[0]) if cls is GenericIcosahedron else plain(cls, seq[0], base=base)
        for s in seq[1:]:
            d.split(s)
        tri = d.get_all_triangles()
        vs = d.get_all_vertices()
        r = {'xyz': d.get_all_xyz(), 'tri': tri, 'xy': np.array([(v.x, v.y) for v in vs]),
             'same': [None if not v.same_vertices else [u.id for u in v.same_vertices] for v in vs],
             'arc': d.arcLength, 'max': (d.x_max, d.y_max), 'freq': d.frequency,
             'latlon': np.array([v.latlon_coord for v in vs[:: max(1, len(vs) // 50)]]),
             'faces_xy': [(v.x, v.y) for v in d.get_faces()],
             'cols': [len(c) for c in d.vertices]}
        nb = []
        for v in vs:
            d.unmark_vertices()
            nb.append([u.id for u in d.get_neighbours(v, False)])
            d.unmark_vertices()
            nb.append([u.id for u in d.get_neighbours(v, True)])
        r['nb'] = nb
        rings = []
        for v in vs[:: max(1, len(vs) // 40)]:
            d.unmark_vertices()
            rings.append([[u.id for u in ring] for ring in d.get_neighbours_in_distance(v, 4)])
        r['rings'] = rings
        out[(base, seq)] = r
    with open(path, 'wb') as fh:
        pickle.dump(out, fh)
    print(f'wrote {len(out)} cases to {path}')


def compare(path_a, path_b):
    with open(path_a, 'rb') as fh:
        a = pickle.load(fh)
    with open(path_b, 'rb') as fh:
        b = pickle.load(fh)
    bad = 0
    for k in a:
        x, y = a[k], b[k]
        msgs = []
        if x['xyz'].shape != y['xyz'].shape:
            msgs.append(f"shape {x['xyz'].shape} {y['xyz'].shape}")
        elif (x['xyz'] != y['xyz']).any():
            msgs.append(f"xyz differs (max {np.abs(x['xyz'] - y['xyz']).max():.2e})")
        for key in ('tri', 'xy', 'latlon'):
            if np.shape(x[key]) != np.shape(y[key]) or not np.allclose(x[key], y[key], atol=1e-15, rtol=0):
                msgs.append(key)
        for key in ('same', 'max', 'freq', 'faces_xy', 'cols', 'nb', 'rings'):
            if x[key] != y[key]:
                msgs.append(key)
        if x['arc'] != y['arc']:
            msgs.append(f"arc {x['arc']} {y['arc']}")
        if msgs:
            bad += 1
            print(k, msgs)
    print('cases', len(a), 'mismatching', bad)
    return bad


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest='cmd', required=True)
p = sub.add_parser('dump')
p.add_argument('out')
p.add_argument('--src')
p = sub.add_parser('compare')
p.add_argument('a')
p.add_argument('b')
args = ap.parse_args()
if args.cmd == 'dump':
    dump(args.out, args.src)
else:
    sys.exit(1 if compare(args.a, args.b) else 0)
