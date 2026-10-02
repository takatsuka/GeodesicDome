# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
3x3 rotation helpers used by the interactive viewer.

Conventions (same as mt.geodesicdome.projection):  latitude = arcsin(z),
longitude = atan2(y, x), so the map centre is the +x axis and north is +z.
A rotation matrix R maps a point p of the original sphere to R @ p in the view.
"""
import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """Rotation about the x axis (the map centre) -- rolls the view."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about the y axis.  A negative angle moves the map centre northwards."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about the z axis (the map's polar axis) -- shifts longitudes by +angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def view_rotation(lat: float, lon: float, roll: float = 0.0) -> np.ndarray:
    """
    Rotation that brings the original-sphere point (lat, lon) [radians] to the map centre,
    then rolls the view by `roll` radians about the centre.
    """
    return rot_x(roll) @ rot_y(lat) @ rot_z(-lon)


def drag_rotation(d_lon: float, d_lat: float) -> np.ndarray:
    """
    Incremental rotation for a mouse drag: `d_lon` spins the sphere about the map's
    polar axis (horizontal drag), `d_lat` tilts it about the east-west axis through the
    map centre (vertical drag).  Apply as  R_new = drag_rotation(...) @ R_old.
    """
    return rot_y(-d_lat) @ rot_z(d_lon)


def orthonormalise(r: np.ndarray) -> np.ndarray:
    """Removes the numerical drift that accumulates after many small rotations."""
    u, _, vt = np.linalg.svd(r)
    return u @ vt


def centre_of_view(r: np.ndarray) -> tuple:
    """(lat, lon) in radians of the original-sphere point currently at the map centre."""
    p = r.T @ np.array([1.0, 0.0, 0.0])
    return float(np.arcsin(np.clip(p[2], -1, 1))), float(np.arctan2(p[1], p[0]))
