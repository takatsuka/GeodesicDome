# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.1] — 2026-10-02

### Added
- **Base polyhedra.** `GeodesicDome(frequency, base=...)` builds the dome on the `'icosahedron'` (default,
  unchanged), the `'tetrahedron'` or the `'dodecahedron'` (also `'icosa'`, `'tetra'`, `'dodeca'`). It returns an
  `IcosahedronDome`, `TetrahedronDome` or `DodecahedronDome`, all subclasses of `GeodesicDome` with the same
  API, index grid and six neighbour offsets.
  - Tetrahedron: the 4HSOM orthogonal array of de Sousa & Oliveira (IJCNN 2012), `(f+1) × (2f+1)` cells,
    2f²+2 points.
  - Dodecahedron: pentakis dodecahedron (each pentagon cut into 5 triangles at its centre) subdivided on the
    flat pentagons, 30f²+2 points, 12 five-neighbour pentagon centres.
  - New module `mt.geodesicdome.grid.polyhedra` with the base solids and their nets (`base_net`,
    `normalise_base`, `BASES`), and a generic grid engine `NetDome` for any such net.
  - `ProjectionViewer(..., colors='base')` colours any dome by the faces of its own base solid
    (`BaseNet.face_centres`).
  - **Interactive example for every dome type**: `examples/11_interactive_base_polyhedra.py` and
    `examples/base_explorer.py` -- the dome on the sphere, on its index grid and in a map centred on a vertex,
    with the vertex's neighbour rings shown in all three (click the net or the map to move it), coloured by
    base face, position or spherical cell area. Notebooks `10_base_polyhedra.ipynb` and
    `11_interactive_base_polyhedra.ipynb` (three map viewers rotated together, the explorer with ipywidgets,
    cell areas at matched size).
  - `examples/10_base_polyhedra.py` and `tests/test_base_polyhedra.py` (index neighbours checked against the
    mesh for every vertex, rings against breadth-first search, the 4HSOM array layout, and the generic engine
    against the icosahedral dome).
- **Example notebooks** in `examples/notebooks/`: one Jupyter notebook per example script (01–09), split into
  explained steps, each ending with ipywidgets controls to explore it (frequency, rings, centre vertex,
  projection, SOM training schedule, export options, viewer rotation, ...). With `ipympl` the figures are live:
  3D plots rotate with the mouse and the `ProjectionViewer` can be dragged. New `notebooks` extra
  (`pip install -e ".[notebooks]"`); `setup_env.sh` installs it by default.

### Fixed
- `SphereMap` (and so `ProjectionViewer`) left thin gaps along the curved map border for coarse meshes
  (e.g. frequency 1-2, or tetrahedral domes): long edges are now drawn through extra points (new
  `max_edge` argument, default 0.2 rad). Faces, face indices and colours are unchanged.

## [1.3.0] — 2026-10-01

### Added
- **`mt.geodesicdome.backend`**: compute backends. `get_backend()` picks the best device automatically:
  NVIDIA GPU through torch (CUDA) or CuPy, the Apple Silicon GPU through torch (MPS), otherwise NumPy on every
  CPU core (work cut into blocks that run on a thread pool; BLAS is multi-threaded). Override with
  `get_backend('cuda' | 'mps' | 'cupy' | 'cpu' | 'numpy' | 'torch:cpu')`, `set_default_backend()` or the
  `MTGEODESIC_BACKEND` environment variable; `MTGEODESIC_NUM_THREADS`, `MTGEODESIC_DTYPE` and `MTGEODESIC_MEMORY`
  tune it. `describe()` lists what is available. torch and CuPy stay optional (`pip install "mtgeodesicdome[gpu]"`).
- **`mt.geodesicdome.compute`**: `DomeArrays.from_dome()` (points, faces, edges, index map, ring length),
  `angular_distance()`, `ring_distance()` and `nearest_vertex()` on the chosen backend, in memory-bounded blocks.

### Changed
- `Projection.build` projects all vertices in one vectorised call (same result, much faster on large domes).

## [1.2.0] — 2026-09-28

### Changed
- **New home.** Published as `mtgeodesicdome` (import `mt.geodesicdome`) from
  [`takatsuka/mtGeodesicDome`](https://github.com/takatsuka/mtGeodesicDome).
- `src/` layout; tests in `tests/`; the old plotly/dash viewers moved to `examples/legacy/`.
- GitHub Actions workflows for tests (Linux, macOS, Windows; Python 3.10–3.14) and PyPI publishing.
- Code tidied to pass `ruff check` (type hints use built-in generics such as `list[int]`, unused
  variables removed). No change in behaviour.

### Added
- `tests/test_geodesicdome.py`: sizes, `split` and neighbour-ring checks.
- User guide in `docs/index.md`.

### Fixed
- The abstract methods of `Manifold` and `Projection` raised `TypeError` (`raise NotImplemented()`);
  they now raise `NotImplementedError`.

---

## Earlier history

### 1.1.0 (first release since 1.0.11 on PyPI)

*Fixes*
* Fixed `GeodesicDome(frequency=n)` for n > 1. Previously it recorded frequency n² and building faces failed.
* Fixed seam linking along one edge of the net (v6–v7 / v10–v7). 2(f−1) vertices there returned only
  4 of their 6 neighbours. Neighbour rings now match graph distance on the sphere for every vertex.
* `latlon_coord` is now stored as float (it was truncated to integers).
* `util.facing` no longer uses `np.cross` on 2D vectors, which is deprecated in NumPy 2.
* Removed debug `print`s from `split()`.

*New*
* Subpackage `interactive`: `ProjectionViewer` (drag to rotate the sphere in a map projection),
  `SphereMap` (seam- and pole-correct projection of a rotated mesh), `unique_mesh` and rotation helpers.
* `Projection` gained vectorised `latlong_to_2d`, `xyz_to_2d_many` and `outline`. Existing methods are unchanged.
* `GeodesicDome.__repr__` and `__version__`.
* The `examples/` folder, this README, a test suite, and `setup_env.sh` for a complete development
  environment.

*Packaging*
* Licensed under **AGPL-3.0-or-later** with an attribution term (see the [README](https://github.com/takatsuka/mtGeodesicDome#8-licence)).
* Built from `pyproject.toml`, which replaces `setup.py`. Correct requirements: Python ≥ 3.10 and NumPy,
  plus optional extras.
* Test scripts are no longer shipped in the wheel.

### 1.0.7 – 1.0.11 (2024): earlier releases.
