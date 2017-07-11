#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sqrt, cos, sin, radians

import numpy as np


def dcos_plane(attitude):
    """Converts poles into direction cossines."""  # bad
    dd, d = np.transpose(np.radians(attitude))  # dip direction, dip
    return np.array((-np.sin(d)*np.sin(dd),
                     -np.sin(d)*np.cos(dd),
                     -np.cos(d))).T


def sphere_plane(data):
    """Calculates the attitude of poles direction cossines."""   # bad
    x, y, z = np.transpose(data)
    sign_z = np.where(z > 0, 1, -1)
    z = np.clip(z, -1., 1.)
    return np.array((np.degrees(np.arctan2(sign_z*x, sign_z*y)) % 360,
                     np.degrees(np.arccos(np.abs(z))))).T


def dcos_line(attitude):
    """Converts the attitude of lines (trend, plunge) into
    direction cosines."""  # OK?
    tr, pl = np.transpose(np.radians(attitude))  # trend, plunge
    return np.array((np.cos(pl)*np.sin(tr),
                    np.cos(pl)*np.cos(tr),
                     -np.sin(pl))).T


def dcos_rake(attitude):
    """Converts the attitude of lines (dip direction, dip, rake) into
    direction cosines."""  # OK?
    dd, d, rk = np.transpose(np.radians(attitude))  # trend, plunge
    return np.array((np.sin(rk)*np.cos(d)*np.sin(dd) - np.cos(rk)*np.cos(dd),
                    np.sin(rk)*np.cos(d)*np.cos(dd) + np.cos(rk)*np.sin(dd),
                     -np.sin(rk)*np.sin(d))).T


def sphere_line(data):
    """Returns the attitude of lines direction cosines."""  # bad
    x, y, z = np.transpose(data)
    sign_z = np.where(z > 0, -1, 1)
    z = np.clip(z, -1., 1.)
    return np.array((np.degrees(np.arctan2(sign_z*x, sign_z*y)) % 360,
                     np.degrees(np.arcsin(np.abs(z))))).T


def project_equal_angle(data, invert_positive=True):
    """equal-angle (stereographic) projection.

    Projects a point from the unit sphere to a plane using
    stereographic projection"""
    x, y, z = np.transpose(data)
    d = 1./np.sqrt(x*x + y*y + z*z)
    if invert_positive:
        c = np.where(z > 0, -1, 1)*d
        x, y, z = c*x, c*y, c*z
    else:
        x, y, z = d*x, d*y, d*z
    return x/(1-z), y/(1-z)


def read_equal_angle(data):
    """inverse equal-angle (stereographic) projection.

    Inverts the projection of a point from the unit sphere
    to a plane using stereographic projection"""
    X, Y = np.transpose(data)
    x = 2.*X/(1. + X*X + Y*Y)
    y = 2.*Y/(1. + X*X + Y*Y)
    z = (-1. + X*X + Y*Y)/(1. + X*X + Y*Y)
    return x, y, z


def project_equal_area(data, invert_positive=True):
    """equal-area (schmidt-lambert) projection.

    Projects a point from the unit sphere to a plane using
    lambert equal-area projection, though shrinking the projected
    sphere radius to 1 from sqrt(2)."""
    x, y, z = np.transpose(data)
    # normalize the data before projection
    d = 1./np.sqrt(x*x + y*y + z*z)
    if invert_positive:
        c = np.where(z > 0, -1, 1)*d
        x, y, z = c*x, c*y, c*z
    else:
        x, y, z = d*x, d*y, d*z
    return x*np.sqrt(1/(1-z)), y*np.sqrt(1/(1-z))


def read_equal_area(data):
    """inverse equal-area (schmidt-lambert) projection.

    Inverts the projection of a point from the unit sphere
    to a plane using lambert equal-area projection, cosidering
    that the projected radius of the sphere was shrunk to 1 from
    sqrt(2)."""
    X, Y = np.transpose(data)*sqrt(2)  # Does python optimize this?
    x = np.sqrt(1 - (X*X + Y*Y)/4.)*X
    y = np.sqrt(1 - (X*X + Y*Y)/4.)*Y
    z = -1. + (X*X + Y*Y)/2
    return x, y, z


def normalized_cross(a, b):
    """Returns the normalized cross product between input vectors."""
    c = np.cross(a, b)
    length = sqrt(c.dot(c))
    return c/length if length > 0 else c

def general_plane_intersection(n_a, da, n_b, db):
    """Returns a point and direction vector for the line of intersection
    of two planes in space, or None if planes are parallel."""
    # https://en.wikipedia.org/wiki/Intersection_curve
    l_v = np.cross(n_a, n_b)
    norm_l = sqrt(l_v.dot(l_v))
    if norm_l == 0:
        return None
    else:
        l_v /= norm_l
    aa = n_a.dot(n_a)
    bb = n_b.dot(n_b)
    ab = n_a.dot(n_b)
    d_ = 1./(aa*bb - ab*ab)
    l_0 = (da*bb - db*ab)*d_*n_a + (db*aa - da*ab)*d_*n_b
    return l_v, l_0


# Should the answers be normalized?
def small_circle_intersection(axis_a, angle_a, axis_b, angle_b):
    """Returns, if exists, the intersection of two small circles."""
    line = general_plane_intersection(axis_a, cos(angle_a),
                                      axis_b, cos(angle_b))
    if not line:
        return None
    l_v, l_0 = line
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    b = 2*l_v.dot(l_0)
    delta = b*b - 4*(l_0.dot(l_0) - 1)
    if delta < 0:
        return None
    elif delta == 0:
        return -b/2.,
    else:
        sqrt_delta = sqrt(delta)
        return l_0 + l_v*(-b - sqrt_delta)/2., l_0 + l_v*(-b + sqrt_delta)/2.

def build_rotation_matrix(azim, plng, rake):
    """Returns the rotation matrix that rotates the axis to the given plane
    with rake."""
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
