# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Build time and live memory of the tetrahedral (f = 64) and dodecahedral (f = 32) domes.

    python benchmarks/bench_bases.py new
    python benchmarks/bench_bases.py newobj
    python benchmarks/bench_bases.py old --src /tmp/gd-old/src
    python benchmarks/bench_bases.py core --src /path/to/geodesic-dome-site

See bench_icosahedron.py for what `old`, `newobj` and `core` mean.
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
args = ap.parse_args()
impl = args.impl
sys.path.insert(0, str(args.src or REPO_SRC))


def plain(cls, *args, **kwargs):
    """The unrelaxed dome (relax=False; older versions without the option are always unrelaxed)."""
    try:
        return cls(*args, relax=False, **kwargs)
    except TypeError:
        return cls(*args, **kwargs)


def make(f, base):
    if impl == 'core':
        from geodesic_grid_core import IndexedGridCore
        return IndexedGridCore(f, base=base, backend='numpy')
    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    d = plain(GeodesicDome, f, base=base)               # build time without relaxation
    if impl == 'newobj':
        d.get_all_vertices()
    return d


print('impl,base,frequency,build_ms,MiB')
for base, f in (('tetrahedron', 64), ('dodecahedron', 32)):
    make(2, base)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        make(f, base)
        ts.append(time.perf_counter() - t)
    gc.collect()
    tracemalloc.start()
    d = make(f, base)
    mem = tracemalloc.get_traced_memory()[0] / 2**20
    tracemalloc.stop()
    print(f'{impl},{base},{f},{statistics.median(ts) * 1e3:.1f},{mem:.2f}')
