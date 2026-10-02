# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABC

import numpy as np
from numpy import array

from mt.geodesicdome.projection.projection import Projection


class EqualEarth(Projection, ABC):
    """
    https://en.wikipedia.org/wiki/Equal_Earth_projection

    """
    def _latlong_to_2d(self, latlon: array) -> array:
        # print(f'lat, long = {latlon[0]}, {latlon[1]}')
        sin_th = np.sin(latlon[0]) * np.sqrt(3) * 0.5
        theta = np.arcsin(sin_th)
        a1 = 1.340264
        a2 = -0.081106
        a3 = 0.000893
        a4 = 0.003796

        x: float = 2 * np.sqrt(3) * latlon[1] * np.cos(theta) / (
            3 * (9*a4*np.power(theta, 8) + 7*a3*np.power(theta, 6) + 3*a2*np.power(theta, 2) + a1))
        y: float = a4 * np.power(theta, 9) + a3 * np.power(theta, 7) + a2 * np.power(theta, 3) + a1 * theta
        # print(f'x, y = {x}, {y}')
        return np.array([x, y])
