#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

direction_pattern = re.compile(b'([NS]?)([0-9.]+)([EW]?).*')
dip_pattern = re.compile(b'([0-9.]+)([NESW]*).*')


def parse_direction(direction):
    try:
        return float(direction)
    except ValueError:
        pass
    try:
        leading, value, trailing = direction_pattern.match(direction).groups()
    except AttributeError:
        raise ValueError("invalid direction: %s" % direction)
    value = float(value)
    if not trailing:
        return value
    elif trailing == "E":
        value *= 1.
    elif trailing == "W":
        value *= -1.
    if leading == "N":
        return value % 360.
    elif leading == "S":
        return 180. - value


def parse_dip(dip):
    try:
        return float(dip), 0.
    except ValueError:
        pass
    try:
        value, quadrant = dip_pattern.match(dip).groups()
    except AttributeError:
        raise ValueError("invalid dip: %s" % dip)
    value = float(value)
    if quadrant in ["NE", "NW"]:
        return value, -90.
    elif quadrant in ["SE", "SW"]:
        return value,  90.
    else:
        raise ValueError("invalid dip: %s" % dip)


def parse_attitude(direction, dip, strike=False):
    theta = parse_direction(direction.upper())
    phi, quad_modifier = parse_dip(dip.upper())
    if quad_modifier == 0. and strike:  # for right hand rule
        quad_modifier = 90.
    elif strike:
        theta %= 180
    return (theta + quad_modifier) % 360, phi

def parse_attitude(direction, dip, strike=False):
    try:
        parsed_direction = float(direction)
    except ValueError:
        try:
            match = direction_pattern.match(direction.upper())
            leading, value, trailing = match.groups()
        except AttributeError:
            raise ValueError("invalid direction: %s" % direction)
        if not value:
            raise ValueError("invalid direction: %s" % direction)
        if trailing and not leading:
            raise ValueError("invalid direction: %s" % direction)
        if trailing == 'W':
            parsed_direction = -float(value)
        else:
            parsed_direction = float(value)
        if leading == 'S':
            parsed_direction = 180. - parsed_direction
    try:
        parsed_dip = float(dip)
    except ValueError:
        try:
            match = dip_pattern.match(dip.upper())
            value, quadrant = match.groups()
        except AttributeError:
            raise ValueError("invalid dip: %s" % dip)
        parsed_dip = float(dip)
        if not quadrant and strike:
            parsed_direction += 90.
        elif quadrant in ["NE", "NW"]:


