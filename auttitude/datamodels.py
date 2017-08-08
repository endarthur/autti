#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from math import acos, asin, atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np

from auttitude.io import sphere_line, sphere_plane
from auttitude.math import normalized_cross
from auttitude.stats import DEFAULT_GRID, SphericalStatistics


class Vector(np.ndarray):
    def __new__(cls, dcos_data):
        return np.asarray(dcos_data).view(cls)

    def angle_with(self, other):
        """Returns the angle (in radians) between both vectors using
        the dot product between them."""
        self_length = self.length
        other_length = sqrt(other.dot(other))
        return acos(self.dot(other) / (self_length * other_length))

    def cross_with(self, other):
        """Returns the cross product between both vectors."""
        return Vector(np.cross(self, other))

    def normalized_cross_with(self, other):  # Is this necessary?
        """Returns the normalized cross product between both vectors."""
        return Vector(normalized_cross(self, other))

    @property
    def attitude(self):
        """Returns the spherical coordinates of the normalized vector,
        considering it to be a Line in geological sense, as a
        Trend/Plunge pair in degrees."""
        x, y, z = self / self.length
        if z > 0:
            x, y = -x, -y
        return degrees(atan2(x, y)) % 360, degrees(asin(abs(z)))

    @property  # this should be cached
    def length(self):
        """Returns the euclidian norm of this vector."""
        return sqrt(self.dot(self))

    @property
    def direction_vector(self):
        """Returns the vector's left horizontal perpendicular vector.
        defaults to (1, 0, 0) if the vector is vertical."""
        if self[2] == 1.:
            return Vector((1., 0., 0.))
        direction = Vector((self[1], -self[0], 0.))
        return direction / direction.length

    @property
    def dip_vector(self):
        """Returns the vector perpendicular to both this vector
        and it's direction vector. If this vector represents a plane,
        the resulting vector represents the maximum slope direction."""
        return Vector(np.cross(self / self.length, self.direction_vector))

    @property
    def projection_matrix(self):
        """Returns the matrix that projects vectors onto this vector."""
        return np.outer(self, self)

    @property
    def rejection_matrix(self):
        """Returns the matrix that rejects of a vector to this vector."""
        return np.eye(3) - self.projection_matrix

    @property
    def cross_product_matrix(self):
        """Returns the matrix that operates the cross product with this vector
        when multiplied by another vector"""
        return np.array(((0., -self[2], self[1]), (self[2], 0., -self[0]),
                         (-self[1], self[0], 0.)))

    def get_rotation_matrix(self, theta):
        """Returns the counterclockwise rotation matrix about this vector
        by angle theta."""
        return cos(theta)*np.eye(3) + sin(theta)*self.cross_product_matrix +\
            (1 - cos(theta))*self.projection_matrix

    def get_great_circle(self, step=radians(1.)):
        """Returns an array of n points equally spaced along the great circle
        normal to this vector."""
        theta_range = np.arange(0, 2 * pi, step)
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self.direction_vector[:, None] * cos_range +
                self.dip_vector[:, None] * sin_range).T,

    def get_small_circle(self, alpha, step=radians(1.)):
        """Retuns a pair of arrays representing points spaced step along
        both small circles with an semi-apical opening of alpha around
        this vector."""
        sc = self.get_great_circle(step)[0].T * sin(alpha) + self[:, None] * cos(
            alpha)
        return sc.T, -sc.T

    def arc_to(self, other, step=radians(1.)):
        """Returns an array of points spaced step along the great circle
        between both vectors."""
        normal = self.rejection_matrix.dot(other)
        normal /= sqrt(normal.dot(normal))
        theta_range = np.arange(0, self.angle_with(other), step)
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self * cos_range[:, None] + normal * sin_range[:, None]),


class Plane(Vector):
    def intersection_with(self, other):
        """Returns the plane containing both lines."""
        line = Line(self.cross_with(other))
        line_length = line.length
        return line / line_length if line_length > 0 else line

    @property
    def attitude(self):
        """Returns the spherical coordinates of the plane as a
        Dip Direction/Dip pair, in degrees."""
        x, y, z = self / self.length
        if z > 0:
            x, y = -x, -y
        return degrees(atan2(-x, -y)) % 360, degrees(acos(abs(z)))


class Line(Vector):
    def plane_with(self, other):
        """Returns the line of intersection of both planes."""
        plane = Plane(self.cross_with(other))
        plane_length = plane.length
        return plane / plane_length if plane_length > 0 else plane


class VectorSet(np.ndarray):
    item_class = Vector

    def __new__(cls, dcos_data):
        obj = np.asarray(dcos_data).view(cls)
        return obj

    def __finalize_array__(self, obj):
        if obj is None:
            return

    def __getitem__(self, x):
        item = super(VectorSet, self).__getitem__(x)
        if np.atleast_2d(item).shape == (1, 3):
            return item.view(self.item_class)
        else:
            return item

    @property
    def stats(self):
        """Contains spherical statstics for the data."""
        return SphericalStatistics(self)

    @property
    def attitude(self):
        """Converts this data from direction cosines to attitudes."""
        return sphere_line(self)

    def count_fisher(self, k=None, grid=None):
        """Performs grid counting of the data by Fisher smoothing."""
        if grid is None:
            grid = DEFAULT_GRID
        return grid.count_fisher(self, k)

    def count_kamb(self, theta=None, grid=None):
        """Performs grid counting of the data by small circles of
        apperture theta."""
        if grid is None:
            grid = DEFAULT_GRID
        return grid.count_kamb(self, theta)

    def normalized_cross_with(self, other):
        """Returns a VectorSet object containing the normalized cross
        product of all possible pairs betweeen this VectorSet and an
        (n, 3) array-like"""
        vectors = np.zeros((len(self) * len(other), 3))
        i = 0
        for self_vector in self:
            for other_vector in other:
                cross = normalized_cross(self_vector, other_vector)
                vectors[i] = cross if cross[2] < 0 else -cross
                i += 1
        return VectorSet(vectors)

    def angle_with(self, other):
        """Returns the angles matrix between this Spherical Data and an
        (n, 3) array-like"""
        angles = np.zeros((len(self), len(other)))
        for i, self_vector in enumerate(self):
            for j, other_vector in enumerate(other):
                angles[i, j] = self_vector.angle_with(other_vector)
        return angles

    def get_great_circle(self, step=radians(1.)):
        """Returns a generator to the great circles of this VectorSet
        vectors."""
        for vector in self:
            yield vector.get_great_circle(step)[0]  # because of plot_circles


class PlaneSet(VectorSet):
    item_class = Plane

    def intersection_with(self, other):
        return self.normalized_cross_with(other).view(LineSet)

    @property
    def attitude(self):
        """Converts this data from direction cosines to attitudes."""
        return sphere_plane(self)


class LineSet(VectorSet):
    item_class = Line

    def planes_with(self, other):
        return self.normalized_cross_with(other).view(PlaneSet)
