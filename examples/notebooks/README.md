# Example notebooks

The example scripts in `examples/` as Jupyter notebooks. Each one walks through its script in small,
explained steps and ends with a **Try it** section: ipywidgets sliders, drop-downs and buttons to explore it.

| Notebook | Shows | Try it |
|---|---|---|
| `01_quickstart.ipynb` | building domes, NumPy export, vertex attributes, sizes per frequency | inspect any vertex; pick a frequency |
| `02_plot_dome_3d.ipynb` | shaded 3D renders of frequency 1, 2, 4 and 8 | frequency, colouring, camera |
| `03_unfolded_net.ipynb` | the index grid, seam copies and the 12 five-neighbour corners | follow a vertex and its neighbours across the seams |
| `04_neighbours.ipynb` | *k*-ring neighbourhoods around interior, seam and corner vertices | frequency, number of rings, centre vertex |
| `05_map_projections.ipynb` | the four projections | projection, frequency, colouring, edges |
| `06_spherical_som.ipynb` | a SOM trained on colours on the sphere, its learning curve, and the trained map in the viewer | train your own (size, samples, radius, seed) |
| `07_export_mesh.ipynb` | writing a de-duplicated mesh to `.obj`, `.off` and `.npz`, and reloading it | export any frequency and radius |
| `08_plane_grids.ipynb` | hexagonal and rectilinear `Plane`, with borders and as a torus | size, lattice, topology, centre, rings |
| `09_interactive_projection.ipynb` | `ProjectionViewer`: rotate the sphere inside a map projection | projection, colouring, centre of view, roll |
| `10_base_polyhedra.ipynb` | domes on the tetrahedron (4HSOM array), icosahedron and dodecahedron: sizes, the tetrahedral array, nets, corner neighbours | frequency, which solids |
| `11_interactive_base_polyhedra.ipynb` | every dome type interactively: three map viewers rotated together, the explorer (sphere + net + map with neighbour rings across seams), cell areas at matched size | base, frequency, rings, colouring, centre (click the net or map when live), view |

## Running them

```bash
pip install -e ".[notebooks]"        # or ./setup_env.sh, which includes it
jupyter lab examples/notebooks
```

They work straight from a checkout: the first cell imports `nb_setup.py` (in this folder), which puts
`src/` and `examples/dome_utils.py` on the path and chooses the matplotlib backend.

* **With `ipympl`** (in the `notebooks` extra) figures are live: drag 3D plots to rotate them, and drag the map
  in notebooks 09 and 11 to rotate the sphere; in notebook 11 click the net or the map to choose a vertex.
* **Without it** figures are static images; the controls still work and redraw them. Set the environment
  variable `MTG_NOTEBOOK_STATIC=1` to force this.

VS Code and PyCharm open the notebooks too; choose `~/.venvs/GeodesicDome/bin/python` as the kernel.

The notebooks are stored without outputs. Unlike the scripts they do not overwrite the showcase images in
`examples/output/`; only notebook 07 writes files there (`.obj`, `.off`, `.npz`, which git ignores).
