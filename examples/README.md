# Examples

Runnable scripts for `mt.geodesicdome`. They work straight from a checkout (`dome_utils.py` puts `src/` on
`sys.path`), and write their images and files to `examples/output/`. See section 5 of the main README for
what each one shows.

```bash
pip install -e ".[examples]"
python examples/01_quickstart.py
python examples/11_interactive_base_polyhedra.py     # tetrahedron / icosahedron / dodecahedron domes, interactively
```

`notebooks/` has the same examples as Jupyter notebooks, with interactive controls
(`pip install -e ".[notebooks]"`, then `jupyter lab examples/notebooks`).

`legacy/` holds the older plotly/dash viewers (`pip install -e ".[legacy]"`). They open windows or start a
web server, so they are not part of the test suite.
