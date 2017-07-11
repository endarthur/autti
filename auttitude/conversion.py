#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

direction_pattern = re.compile(b'([NS]?)([0-9.]*)([EW]?).*')
dip_pattern = re.compile(b'([0-9.]*)([NESW]*).*')


def translate_attitude(direction, dip, strike=False):
    try:
        dip = float(dip)
        dip_quadrant = ""
    except ValueError:
        dip, dip_quadrant = dip_pattern.match(dip.upper()).groups()
        dip = float(dip)
        if dip_quadrant not in ("NE", "SE", "SW", "NW"):
            raise ValueError("invalid dip quadrant in: %s" % dip)
    try:
        direction = float(direction)
    except ValueError:
        leading, value, trailing = direction_pattern.match(
            direction.upper()).groups()
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
