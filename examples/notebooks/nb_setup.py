# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Shared set-up for the example notebooks in this folder.

    import nb_setup
    WIDGET = nb_setup.setup()        # True when figures are live (ipympl), False when static

`setup()` makes `mt.geodesicdome` and `examples/dome_utils.py` importable from a source
checkout (no `pip install` needed) and picks the matplotlib backend:

* with `ipympl` installed  -> `%matplotlib widget`: figures are live, 3D plots rotate with the
  mouse and the ProjectionViewer can be dragged;
* otherwise                -> `%matplotlib inline`: static images, the sliders still work.

`LiveFigure` is a figure that the ipywidgets controls redraw in place under either backend.
"""
import os
import sys

try:
    import numpy  # noqa: F401
except ImportError:
    raise ImportError(
        f'numpy is not installed for the Python running this notebook:\n    {sys.executable}\n'
        'Run ./setup_env.sh in the repository once, then choose ~/.venvs/mtGeodesicDome/bin/python as the '
        'interpreter (PyCharm: Settings > Project > Python Interpreter; VS Code / Jupyter: the kernel picker).'
    ) from None

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

import dome_utils  # noqa: E402, F401  (adds ../../src to sys.path)

WIDGET = False


def setup(interactive: bool = True) -> bool:
    """Chooses the matplotlib backend; returns True when figures are live (ipympl)."""
    global WIDGET
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:                                       # plain python, not a notebook
        ipython = None
    if ipython is None:
        return False

    WIDGET = False
    if interactive and not os.environ.get('MTG_NOTEBOOK_STATIC'):
        try:
            import ipympl  # noqa: F401
            ipython.run_line_magic('matplotlib', 'widget')
            WIDGET = True
        except ImportError:
            pass
    if not WIDGET:
        ipython.run_line_magic('matplotlib', 'inline')
    print('figures: live (ipympl) -- drag 3D plots to rotate them' if WIDGET else
          'figures: static (inline) -- `pip install ipympl` for live, rotatable figures')
    return WIDGET


class LiveFigure:
    """
    A figure that interactive controls redraw in place.

        live = LiveFigure(figsize=(8, 6))

        @interact(freq=(1, 12))
        def draw(freq):
            fig = live.clear()              # an empty figure to draw on
            ...
            live.refresh()

        live.show()                         # put the (live) canvas under the controls

    :param fig: wrap an existing figure (e.g. `ProjectionViewer(...).fig`) instead of making one
    """

    def __init__(self, fig=None, **figure_kwargs):
        import matplotlib.pyplot as plt
        if fig is None:
            with plt.ioff():                                  # do not auto-display it
                fig = plt.figure(**figure_kwargs)
        self.fig = fig
        if not WIDGET:
            plt.close(fig)                                    # inline: shown only via refresh()

    def clear(self):
        self.fig.clear()
        return self.fig

    def refresh(self):
        """Redraws: in place when live, or as a new image in the current output when static."""
        if WIDGET:
            self.fig.canvas.draw_idle()
        else:
            from IPython.display import display
            display(self.fig)

    def show(self):
        """Displays the live canvas (call once, at the end of the cell). No-op when static."""
        if WIDGET:
            from IPython.display import display
            display(self.fig.canvas)
