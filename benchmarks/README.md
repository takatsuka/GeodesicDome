# Benchmarks

Scripts and results behind [docs/performance.md](../docs/performance.md), the write-up of the
array-first GeodesicDome rewrite.

| File | What it does |
|---|---|
| `bench_icosahedron.py` | Build time, live memory and hop-search times of the icosahedral dome |
| `bench_bases.py` | Build time and memory of the tetrahedral (f = 64) and dodecahedral (f = 32) domes |
| `check_equivalence.py` | Dumps two implementations and checks they give identical domes (44 cases) |
| `check_hops.py` | Checks `within_hops`, `neighbour_ids` and `within_arc` against the vertex API |
| `results/*.csv` | The measurements reported in docs/performance.md |
| `results.xlsx` | The same results with summary sheets and speed-up formulas |

Implementations compared:

- `new` — this checkout, arrays only (no vertex objects created)
- `newobj` — this checkout, with every vertex object created
- `old` — the implementation before the rewrite (git `6f664da`)
- `core` — the standalone port `geodesic_grid_core.py` (`IndexedGridCore`), not part of this repository

## Running

```bash
# pre-rewrite source, for `old`
git archive 6f664da src | tar -x -C /tmp/gd-old    # mkdir -p /tmp/gd-old first

python benchmarks/bench_icosahedron.py new --freqs 4 8 16 32 64
python benchmarks/bench_icosahedron.py newobj --freqs 4 8 16 32 64
python benchmarks/bench_icosahedron.py old --src /tmp/gd-old/src --freqs 4 8 16 32 64
python benchmarks/bench_icosahedron.py core --src /path/to/geodesic-dome-site --freqs 4 8 16 32 64

python benchmarks/bench_bases.py new          # likewise newobj / old / core

python benchmarks/check_equivalence.py dump /tmp/old.pkl --src /tmp/gd-old/src
python benchmarks/check_equivalence.py dump /tmp/new.pkl
python benchmarks/check_equivalence.py compare /tmp/old.pkl /tmp/new.pkl

python benchmarks/check_hops.py
```

The recorded results were measured on 7 Oct 2026 in a Linux VM (Python 3.10, NumPy 2.2.6);
absolute times vary by machine, the ratios much less.
