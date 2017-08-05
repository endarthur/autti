#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re

import numpy as np

_DIRECTION_PATTERN = re.compile(r'([NS]?)([0-9.]*)([EW]?).*')
_DIP_PATTERN = re.compile(r'([0-9.]*)([NESW]*).*')


def process_dip(dip_value):
    try:
        dip = float(dip_value)
        dip_quadrant = ""
    except ValueError:
        dip, dip_quadrant = _DIP_PATTERN.match(dip_value.upper()).groups()
        dip = float(dip)
        if dip_quadrant not in ("NE", "SE", "SW", "NW"):
            raise ValueError("invalid dip quadrant in: %s" % dip)
    return dip, dip_quadrant


def process_direction(direction_value):
    try:
        direction = float(direction_value)
    except ValueError:
        leading, value, trailing = _DIRECTION_PATTERN.match(
            direction_value.upper()).groups()
        value = float(value)
        if leading == "N":
            if not trailing or trailing == "E":
                direction = value
            elif trailing == "W":
                direction = 360 - value
        elif leading == "S":
            if trailing == "E":
                direction = 180 - value
            elif trailing == "W":
                direction = 180 + value
            elif not trailing:
                raise ValueError("Invalid direction: %s" % direction)
    return direction


def translate_attitude(direction, dip, strike=False):
    dip, dip_quadrant = process_dip(dip)
    direction = process_direction(direction)
    if dip_quadrant:
        direction = direction % 180.
        if 0 <= direction < 90.:
            if strike:
                if dip_quadrant == "SE":
                    direction = direction + 90.
                elif dip_quadrant == "NW":
                    direction = direction + 270.
            else:
                if dip_quadrant == "NE":
                    pass  # direction = direction
                elif dip_quadrant == "SW":
                    direction = direction + 180.
        else:  # 90 <= direction < 180
            if strike:
                if dip_quadrant == "NE":
                    direction = direction - 90.
                elif dip_quadrant == "SW":
                    direction = direction + 90.
            else:
                if dip_quadrant == "SE":
                    pass  # direction = direction
                elif dip_quadrant == "NW":
                    direction = direction + 180.
    elif strike:  # Right Hand Rule
        direction = (direction + 90.) % 360.

    return direction, dip

def dcos_plane(direction_dip):
    """Converts poles into direction cossines."""  # bad
    dd, d = np.transpose(np.radians(direction_dip))  # dip direction, dip
    return np.array((-np.sin(d)*np.sin(dd),
                     -np.sin(d)*np.cos(dd),
                     -np.cos(d))).T


def sphere_plane(dcos_data):
    """Calculates the attitude of poles direction cossines."""   # bad
    x, y, z = np.transpose(dcos_data)
    sign_z = np.where(z > 0, 1, -1)
    z = np.clip(z, -1., 1.)
    return np.array((np.degrees(np.arctan2(sign_z*x, sign_z*y)) % 360,
                     np.degrees(np.arccos(np.abs(z))))).T


def dcos_line(trend_plunge):
    """Converts the attitude of lines (trend, plunge) into
    direction cosines."""  # OK?
    tr, pl = np.transpose(np.radians(trend_plunge))  # trend, plunge
    return np.array((np.cos(pl)*np.sin(tr),
                     np.cos(pl)*np.cos(tr),
                     -np.sin(pl))).T


def dcos_rake(direction_dip_rake):
    """Converts the attitude of lines (dip direction, dip, rake) into
    direction cosines."""  # OK?
    dd, d, rk = np.transpose(np.radians(direction_dip_rake))  # trend, plunge
    return np.array((np.sin(rk)*np.cos(d)*np.sin(dd) - np.cos(rk)*np.cos(dd),
                     np.sin(rk)*np.cos(d)*np.cos(dd) + np.cos(rk)*np.sin(dd),
                     -np.sin(rk)*np.sin(d))).T


def sphere_line(dcos_data):
    """Returns the attitude of lines direction cosines."""  # bad
    x, y, z = np.transpose(dcos_data)
    sign_z = np.where(z > 0, -1, 1)
    z = np.clip(z, -1., 1.)
    return np.array((np.degrees(np.arctan2(sign_z*x, sign_z*y)) % 360,
                     np.degrees(np.arcsin(np.abs(z))))).T
