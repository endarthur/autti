#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
from __future__ import absolute_import
from math import acos, cos, pi, radians, sin, sqrt
import auttitude as at
import numpy as np


def normalized_cross(a, b):
    """
    Returns the normalized cross product between vectors.
    Uses numpy.cross().
    
    Parameters:
        a: First vector.
        b: Second vector.
    """
    c = np.cross(a, b)
    length = sqrt(c.dot(c))
    return c/length if length > 0 else c


def general_plane_intersection(n_a, da, n_b, db):
    """
    Returns a point and direction vector for the line of intersection
    of two planes in space, or None if planes are parallel.
    
    Parameters:
        n_a: Normal vector to plane A
         da: Point of plane A
        n_b: Normal vector to plane B
         db: Point of plane B
    """
    
    # https://en.wikipedia.org/wiki/Intersection_curve
    
    n_a = np.array(n_a)
    n_b = np.array(n_b)
    da  = np.array(da)
    db  = np.array(db)
    
    l_v = np.cross(n_a, n_b)
    norm_l = sqrt(np.dot(l_v, l_v))
    if norm_l == 0:
        return None
    else:
        l_v /= norm_l
    aa = np.dot(n_a, n_a)
    bb = np.dot(n_b, n_b)
    ab = np.dot(n_a, n_b)
    d_ = 1./(aa*bb - ab*ab)
    l_0 = (da*bb - db*ab)*d_*n_a + (db*aa - da*ab)*d_*n_b
    
    return l_v, l_0


def small_circle_intersection(axis_a, angle_a, axis_b, angle_b):
    """
    Finds the intersection between two small-circles returning zero, one or two 
    solutions as tuple.  
    
    Parameters:
         axis_a: Vector defining first circle axis
        angle_a: Small circle aperture angle (in radians) around axis_a 
         axis_b: Vector defining second circle axis
        angle_b: Small circle aperture angle (in radians) around axis_b
    """
    line = general_plane_intersection(axis_a, cos(angle_a),
                                      axis_b, cos(angle_b))
    if line is None:
        return ()
    l_v, l_0 = line
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    b = 2*l_v.dot(l_0)
    delta = b*b - 4*(l_0.dot(l_0) - 1)
    # Should the answers be normalized?
    if delta < 0:
        return ()
    elif delta == 0:
        return -b/2.,
    else:
        sqrt_delta = sqrt(delta)
        return l_0 + l_v*(-b - sqrt_delta)/2., l_0 + l_v*(-b + sqrt_delta)/2.


def build_rotation_matrix(azim, plng, rake):
    """
    Returns the rotation matrix that rotates the North vector to the line given 
    by Azimuth and Plunge and East and Up vectors are rotate clock-wise by Rake
    around the rotated North vector. 
    
    Parameters:
        azim: Line Azimuth from North (degrees).
        plng: Line Plunge measured from horizontal (degrees).
        rake: Rotation angle around rotated axis (degrees).
    """
    # pylint: disable=bad-whitespace
    azim, plng, rake = radians(azim), radians(plng), radians(rake)

    R1 = np.array((( cos(rake), 0.,        sin(rake)),
                   ( 0.,        1.,         0.      ),
                   (-sin(rake), 0.,        cos(rake))))

    R2 = np.array((( 1.,        0.,        0.       ),
                   ( 0.,        cos(plng), sin(plng)),
                   ( 0.,       -sin(plng), cos(plng))))

    R3 = np.array((( cos(azim), sin(azim), 0.       ),
                   (-sin(azim), cos(azim), 0.       ),
                   ( 0.,        0.,        1.       )))

    return R3.dot(R2).dot(R1)


def adjust_lines_to_planes(lines, planes):
    """
    Project each given line to it's respective plane. Returns the projected
    lines as a new LineSet and the angle (in radians) between each line and
    plane prior to projection.
    
    Parameters: 
        lines:  A LineSet like object with an array of n Lines
        planes: A PlaseSet like object with an array of n Planes 
    """
    
    lines  = at.LineSet(lines)
    planes = at.PlaneSet(planes)
    
    angles = np.zeros(len(lines))
    adjusted_lines = np.zeros_like(lines)
    for i, (line, plane) in enumerate(zip(lines, planes)):
        cos_theta = np.dot(line, plane)
        angles[i] = pi/2. - acos(cos_theta)
        adjusted_line = line - line*cos_theta
        adjusted_lines[i] = adjusted_line/sqrt(np.dot(adjusted_line,
                                                      adjusted_line))
    return adjusted_lines, angles
