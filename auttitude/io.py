#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re

import numpy as np

_DIRECTION_PATTERN = re.compile(r'([NS]?)([0-9.]*)([EW]?).*')
_DIP_PATTERN = re.compile(r'([0-9.]*)([NESW]*).*')


def process_dip(dip_value):
    """
    Parse dip values from string or float. Dip is defined as float number
    or string containing a number followed by two characters indicating the
    quadrant of dip that must match one of NE, SE, SW or NW.
    
    Quadrant of dip should be combined with with direction for proper 
    interpretation of attitude. Please refer to the translate_attitude() method.
    
    Returns the dip value and quadrant.
    
    Parameters:
        dip_value: A string or float containing dip values to be parsed.
    """
    try:
        dip = float(dip_value)
        dip_quadrant = ""
    except ValueError:
        dip, dip_quadrant = _DIP_PATTERN.match(dip_value.strip().upper()).groups()
        dip = float(dip)
        if dip_quadrant not in ("NE", "SE", "SW", "NW"):
            raise ValueError("invalid dip quadrant in: %s" % dip)
    return dip, dip_quadrant


def process_direction(direction_value):
    """
    Parse direction value from a Direction String or Float value. Floats are
    angles from North in clock-wise direction. Direction string encodes an 
    North or South (N or S), a float number and finally an optional East or West
    (E or W).
    
    Returns a direction value float number.
     
    Parameters:
        direction_value: a direction string or float value to be parsed.
    """
    try:
        direction = float(direction_value)
    except ValueError:
        leading, value, trailing = _DIRECTION_PATTERN.match(
            direction_value.strip().upper()).groups()
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
        else:
            raise ValueError("Invalid direction: %s" % direction)
            # should these methods be more or less permissive>?
    return direction


def translate_attitude(direction, dip, strike=False):
    """
    Translate attitude values into proper direction and dip values using special
    methods for parsing spherical oriented notation for translation. Inside
    Auttitude dip is measured along direction unless stated (see strike option).
    
    Parameters:
        direction: A float or string value that can be interpreted as direction.
                   Please refer to io.parse_direction() method.

              dip: A float or string value that can be interpreted as dip.
                   Please refer to io.parse_dip() method.

           strike: If strike is true, this means that the direction GIVEN is not 
                   the direction of dipping, but, direction should be corrected
                   using the dipping quadrant or, if dipping quadrant is no given
                   should use the right-hand-rule meaning that attitude
                   direction (or direction of dipping) is 90 degrees further
                   (clock-wise) than indicated direction.
    """
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
    """
    Converts planes attitude (direction and dip measured from
    horizontal along direction) representing it as direction cosines of plane
    pole as used internally by Auttitude. Direction cosine values have norm
    equal to 1.
    
    Parameters:
        direction_dip: (2,N) iterable elements that contains values of plane
                       azimuth and dip measured from horizontal of N-planes to be
                       represented by its poles direction cosine. 
    """
    dd, d = np.transpose(np.radians(direction_dip))  # dip direction, dip
    return np.array((-np.sin(d)*np.sin(dd),
                     -np.sin(d)*np.cos(dd),
                     -np.cos(d))).T


def sphere_plane(dcos_data, rhr=False):
    """
    Converts to attitude of planes represented by its poles direction cosines.
    Attitudes on those cases are direction measured from North and Dip measured
    along direction unless rhr is set to True.
    
    Parameters:
        dcos_data: (3,N) direction cosines iterable element representing N-plane
                   poles direction cosines.

              rhr: Boolean indicating that direction should be corrected
                   considering the right-hand-rule.
    """
    x, y, z = np.transpose(dcos_data)
    sign_z = np.where(z > 0, 1, -1)
    z = np.clip(z, -1., 1.)
    
    corr = 90.0 if rhr else 0.0
    return np.array((
                     ((np.degrees(np.arctan2(sign_z*x, sign_z*y)) - corr) % 360),
                      np.degrees(np.arccos(np.abs(z))))).T


def dcos_line(trend_plunge):
    """
    Converts the attitude of lines (trend, plunge) into direction cosines values
    as used internally by Auttitude. Direction cosine values have norm equal to
    1.
    
    Parameters:
        trend_plunge: (2,N) elements iterable object that contains values for
                      line trend (orientation from North) and plunge (dipping
                      direction from horizontal) for N-lines to be represented
                      as directional cosines.
    """
    tr, pl = np.transpose(np.radians(trend_plunge))  # trend, plunge
    return np.array((np.cos(pl)*np.sin(tr),
                     np.cos(pl)*np.cos(tr),
                     -np.sin(pl))).T


def dcos_rake(direction_dip_rake):
    """
    Convert lines attitude (dip direction, dip and rake) into direction cosine
    values as used internally by Auttitude. Direction cosine values have norm
    equal to 1.
    
    Parameters:
        direction_dip_rake: (3,N) iterable elements that contains values for
                            direction, dip and rake. Dip is measured from 
                            Horizontal along Direction and Rake is measured
                            from direction.
    """
    dd, d, rk = np.transpose(np.radians(direction_dip_rake))  # trend, plunge
    return np.array((np.sin(rk)*np.cos(d)*np.sin(dd) - np.cos(rk)*np.cos(dd),
                     np.sin(rk)*np.cos(d)*np.cos(dd) + np.cos(rk)*np.sin(dd),
                     -np.sin(rk)*np.sin(d))).T


def sphere_line(dcos_data):
    """
    Returns the attitude of lines from its direction cosines. Lines attitude 
    are defined as trend and plunge values.
    
    Parameters:
        dcos_data: (3,N) direction cosines iterable element representing N-line
                   direction cosines.
    """
    x, y, z = np.transpose(dcos_data)
    sign_z = np.where(z > 0, -1, 1)
    z = np.clip(z, -1., 1.)
    return np.array((np.degrees(np.arctan2(sign_z*x, sign_z*y)) % 360,
                     np.degrees(np.arcsin(np.abs(z))))).T
