# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Checks the index API (within_hops, neighbour_ids, within_arc) against the vertex API.

    python benchmarks/check_hops.py

For every base and several frequencies, ~30 random positions and 0, 1, 2, 5 and 40 hops:
within_hops must reach the same set of sphere points as get_neighbours_in_distance, once each.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mt.geodesicdome.grid.geodesicdome import GeodesicDome  # noqa: E402

rng = np.random.default_rng(1)
n = 0
for base in ('icosahedron', 'tetrahedron', 'dodecahedron'):
    for seq in ((1,), (2,), (3,), (5,), (8,), (2, 3), (13,)):
        d = GeodesicDome(seq[0], base=base)
        for s in seq[1:]:
            d.split(s)
        cls = d.point_index
        N = len(cls)
        assert d.n_points == len(set(cls.tolist()))
        vs = d.get_all_vertices()
        xyz = d.get_all_xyz()
        for i in rng.choice(N, min(N, 30), replace=False).tolist():
            for hops in (0, 1, 2, 5, 40):
                d.unmark_vertices()
                rings = d.get_neighbours_in_distance(vs[i], hops) if hops else [[]]
                want = {cls[i]} | {cls[u.id] for r in rings[:hops] for u in r}
                got = d.within_hops(i, hops)
                assert got[0] == i and len(got) == len(set(cls[got].tolist())), (base, seq, i, hops)
                assert set(cls[got].tolist()) == want, (base, seq, i, hops)
                assert set(cls[d.within_hops(i, hops, include_self=False)].tolist()) == want - {cls[i]}
                n += 1
            d.unmark_vertices()
            assert set(cls[d.neighbour_ids(i)].tolist()) == {cls[u.id] for u in d.get_neighbours(vs[i])}
            near = set(np.flatnonzero(xyz @ xyz[i] >= np.cos(0.3) - 1e-14).tolist())
            assert set(d.within_arc(i, 0.3).tolist()) == near
print(f'{n} hop checks passed')
