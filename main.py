# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Quick demo for running from PyCharm (the green Run button) or `python main.py`.

Builds a geodesic dome, prints its size, and opens the interactive map viewer
(drag to rotate the sphere).  Pass --no-gui to skip the window.

This file is for development only; it is not part of the installed package.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mt.geodesicdome import __version__  # noqa: E402
from mt.geodesicdome.grid.geodesicdome import GeodesicDome  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--freq", type=int, default=8, help="dome frequency (default 8)")
    parser.add_argument("--projection", default="Equal Earth",
                        help="Equal Earth, Kavrayskiy VII, Wagner VI or Wagner III")
    parser.add_argument("--no-gui", action="store_true", help="print the summary only")
    args = parser.parse_args()

    dome = GeodesicDome(args.freq)
    xyz = dome.get_all_xyz()
    triangles = dome.get_all_triangles().reshape(-1, 3)
    print(f"mt.geodesicdome {__version__}")
    print(f"{dome}: {10 * args.freq**2 + 2} points, {len(triangles)} triangles "
          f"({len(xyz)} stored vertices incl. seam copies)")

    if not args.no_gui:
        from mt.geodesicdome.interactive import ProjectionViewer  # needs matplotlib

        ProjectionViewer(dome, args.projection).show()


if __name__ == "__main__":
    main()
