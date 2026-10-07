# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Build time, live memory and hop-search times of the icosahedral GeodesicDome.

    python benchmarks/bench_icosahedron.py new                      # this checkout (arrays only)
    python benchmarks/bench_icosahedron.py newobj                   # this checkout, all vertex objects created
    python benchmarks/bench_icosahedron.py old --src /tmp/gd-old/src
    python benchmarks/bench_icosahedron.py core --src /path/to/geodesic-dome-site
    python benchmarks/bench_icosahedron.py new --freqs 4 8 16 32 64

`old` = the implementation before the array-first rewrite (git 6f664da):
    git archive 6f664da src | tar -x -C /tmp/gd-old
`core` = the standalone port geodesic_grid_core.py (IndexedGridCore).

Build = median of 3 constructions.  Memory = tracemalloc live allocation after construction
("MiB"), and after every lazy structure is built ("MiBall": vertex objects, neighbour table,
hop index).  Queries = median of 80 runs from the middle stored position (20 for the whole
sphere, 3f hops).  `old` / `newobj` search with get_neighbours_in_distance and reset the
visited flags they touched; `core` / `new` use within_hops.
"""
import argparse
import gc
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / 'src'

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('impl', choices=['old', 'new', 'newobj', 'core'])
ap.add_argument('--src', help='source directory to import from (default: this checkout)')
ap.add_argument('--freqs', type=int, nargs='+', default=[4, 8, 16, 32])
args = ap.parse_args()
impl = args.impl
sys.path.insert(0, str(args.src or REPO_SRC))


def make(f):
    if impl == 'core':
        from geodesic_grid_core import IndexedGridCore
        return IndexedGridCore(f, backend='numpy')
    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    d = GeodesicDome(f)
    if impl == 'newobj':
        d.get_all_vertices()
    return d


def med(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def query(d, r):
    if impl == 'core':
        i = len(d) // 2
        return lambda: d.within_hops(i, r)
    if impl == 'new':
        i = len(d.get_all_xyz()) // 2
        return lambda: d.within_hops(i, r)
    vs = d.get_all_vertices()
    v = vs[len(vs) // 2]

    def q():
        rings = d.get_neighbours_in_distance(v, r)
        for u in [v] + [u for ring in rings for u in ring]:
            u.visited = False
            for s in (u.same_vertices or ()):
                s.visited = False
        return rings
    d.unmark_vertices()
    return q


make(2)
print('impl,frequency,build_ms,MiB,MiB_all,r1_ms,r3_ms,r6_ms,full_ms')
for f in args.freqs:
    b = med(lambda f=f: make(f), 3)
    gc.collect()
    tracemalloc.start()
    d = make(f)
    mem = tracemalloc.get_traced_memory()[0] / 2**20
    tracemalloc.stop()
    gc.collect()
    tracemalloc.start()
    d2 = make(f)
    if impl != 'core':
        d2.get_all_vertices()
        d2.unmark_vertices()
        d2.get_neighbours_in_distance(d2.get_all_vertices()[5], 2)
        if impl != 'old':
            d2.within_hops(5, 2)
    mem_all = tracemalloc.get_traced_memory()[0] / 2**20
    tracemalloc.stop()
    del d2
    rs = [med(query(d, r), 80) for r in (1, 3, 6)]
    full = med(query(d, 3 * f), 20)
    print(f'{impl},{f},{b:.2f},{mem:.2f},{mem_all:.2f},{rs[0]:.4f},{rs[1]:.4f},{rs[2]:.4f},{full:.3f}')
