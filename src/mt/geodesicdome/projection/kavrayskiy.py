# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABC

import numpy as np
from numpy import array

from mt.geodesicdome.projection.projection import Projection


class KavrayskiyVII(Projection, ABC):
    def _latlong_to_2d(self, latlon: array) -> array:
        # print(f'lat, long = {latlon[0]}, {latlon[1]}')
        x: float = 3 * latlon[1] * 0.5 * np.sqrt(1 / 3 -latlon[0] * latlon[0] / (np.pi * np.pi))
        y: float = latlon[0]
        # print(f'x, y = {x}, {y}')
        return np.array([x, y])
