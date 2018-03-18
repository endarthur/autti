#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from math import acos, asin, atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np

from auttitude.io import sphere_line, sphere_plane, translate_attitude, dcos_plane, dcos_line
from auttitude.math import normalized_cross
from auttitude.stats import DEFAULT_GRID, SphericalStatistics


class Vector(np.ndarray):
    """
    Class that represents one normalized vector in space. This class 
    extends Numpy.ndarray class that is used as storage container for 
    the information.

    Parameters:
        dcos_data: Iterable object with 3 elements (can be even another 
        Vector) to construct a Vector from. Inside 'io' module there
        are methods to convert attitude data to normalized direction 
        cosines.
    """

    def __new__(cls, dcos_data):
        return np.asarray(dcos_data).view(cls)

    def angle_with(self, other):
        """Returns the angle (in radians) between both vectors using
        the dot product between them.

        Parameter:
            other: A Vector like object.
        """
        self_length = self.length
        other_length = sqrt(other.dot(other))
        return acos(self.dot(other) / (self_length * other_length))

    def cross_with(self, other):
        """Returns the cross product between both vectors.

        Parameter:
            other: A Vector like object.
        """
        return Vector(np.cross(self, other))

    def normalized_cross_with(self, other):
        """Returns the normalized cross product between this and other
        vectors.

        Parameter:
            other: A Vector like object.
        """
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
        by angle theta.

        Parameters:
            theta: Rotation angle in radians
        """
        return cos(theta)*np.eye(3) + sin(theta)*self.cross_product_matrix +\
            (1 - cos(theta))*self.projection_matrix

    def get_great_circle(self, step=radians(1.)):
        """Returns an array of n points equally spaced along the great circle
        normal to this vector.

        Parameters:
            step: Angular step in radians to generate points around great 
            circle.
        """
        theta_range = np.arange(0, 2 * pi, step)
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self.direction_vector[:, None] * cos_range +
                self.dip_vector[:, None] * sin_range).T,

    def get_small_circle(self, alpha, A=0, B=0, step=radians(1.)):
        """Returns a pair of arrays representing points spaced step along
        both small circles with an semi-apical opening of alpha around
        this vector.

        Parameters:
            alpha: Apperture of the small circle in radians
            
            step: Angular step in radians to generate points around small
            circle.
        """
        if A == 0 and B == 0:
            sc = self.get_great_circle(step)[0].T * sin(
                alpha) + self[:, None] * cos(alpha)
        else:
            theta_range = np.arange(0, 2*pi, step)
            alpha_ = alpha + A * np.cos(2*theta_range) + B * np.sin(2*theta_range)
            sc = self.get_great_circle(step)[0].T * np.sin(
                alpha_) + self[:, None] * np.cos(alpha_)
        return sc.T, -sc.T

    def arc_to(self, other, step=radians(1.)):
        """Returns an array of points spaced step along the great circle
        between both vectors.

        Parameters:
            step: Angular step in radians to generate points along the 
            great-circle arc.
        """
        normal = self.rejection_matrix.dot(other)
        normal /= sqrt(normal.dot(normal))
        theta_range = np.arange(0, self.angle_with(other), step)
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self * cos_range[:, None] + normal * sin_range[:, None]),


class Plane(Vector):
    """
    Like the Vector class but, more specifically representing a plane in 
    space defined by the direction cosines of the plane dip direction/dip 
    pair.

    Parameters:
        dcos_data: Direction cosines of the dip direction/dip pair.
    """

    @staticmethod
    def from_attitude(direction, dip, strike=False):
        """
        Return a new Plane from direction, dip and strike given.
        Please refer to translate_attitude method for  parameters description.
        """
        dd, d = translate_attitude(direction, dip, strike)
        return Plane(dcos_plane((dd, d)))

    def intersection_with(self, other):
        """Returns the line of intersection of this and the other plane.

        Parameter:
            other: a Plane like object that will intersect with this object.
        """
        line = Line(self.cross_with(other))
        line_length = line.length
        return line / line_length if line_length > 0 else line

    @property
    def rhr_attitude(self):
        dd, d = self.attitude
        return (dd - 90) % 360, d

    @property
    def attitude(self):
        """Returns the spherical coordinates of the plane as a
        Dip Direction/Dip pair, in degrees."""
        x, y, z = self / self.length
        if z > 0:
            x, y = -x, -y
        return degrees(atan2(-x, -y)) % 360, degrees(acos(abs(z)))


class Line(Vector):
    """
    Like the Vector class but, more specifically representing a line in 
    space defined by the direction cosines of the line direction/dip.

    Parameters:
        dcos_data: Direction cosines of the line direction/dip.
    """

    @staticmethod
    def from_attitude(direction, dip, strike=False):
        """
        Return a new Line Object from direction, dip and strike given.
        Please refer to translate_attitude method for description of parameters.
        """
        direction, dip = translate_attitude(direction, dip, strike)
        return Line(dcos_line((direction, dip)))

    def plane_with(self, other):
        """Returns the plane containing this and the other line.

        Parameter:
            other: a Line like object that will define the returned plane.
        """
        plane = Plane(self.cross(other))
        plane_length = plane.length
        return plane / plane_length if plane_length > 0 else plane


class VectorSet(np.ndarray):
    """Class that represents a set (collection) of Vectors.

    Parameters:
        dcos_data: Is an array of direction cosines.
    """
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
        """Contains spherical statistics object for the data
        set.
        """
        return SphericalStatistics(self)

    @property
    def attitude(self):
        """Converts this data from direction cosines to attitudes."""
        return sphere_line(self)

    def count_fisher(self, k=None, grid=None):
        """Performs grid counting of the data by Fisher smoothing.

        Parameters:
            k: von Mises-Fisher k parameter, see 
            stats.SphericalGrid.count_fisher.
            
            grid: A stats.Spherical grid object to count on. If None
            the default grid defined on stats.DEFAULT_GRID will be
            used.
        """
        if grid is None:
            grid = DEFAULT_GRID
        return grid.count_fisher(self, k)

    def count_kamb(self, theta=None, grid=None):
        """Performs grid counting of the data by small circles of
        aperture theta.

        Parameters:
            theta: Robin and Jowett (1986) based on Kamb (1956) theta
            parameter, see stats.SphericalGrid.count_kamb.
            
            grid: A stats.Spherical grid object to count on. If None
            the default grid defined on stats.DEFAULT_GRID will be
            used.
        """
        if grid is None:
            grid = DEFAULT_GRID
        return grid.count_kamb(self, theta)

    def normalized_cross_with(self, other):
        """Returns a VectorSet object containing the normalized cross
        product of all possible pairs between this VectorSet and an
        (n, 3) array-like

        Parameter:
            other: A VectorSet like object.
        """
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
        (n, 3) array-like

        Parameter:
            other: A VectorSet like object.
        """
        angles = np.zeros((len(self), len(other)))
        for i, self_vector in enumerate(self):
            for j, other_vector in enumerate(other):
                angles[i, j] = self_vector.angle_with(other_vector)
        return angles

    def get_great_circle(self, step=radians(1.)):
        """Returns a generator to the list of great circles of 
        this VectorSet vectors.

        Parameters:
            step: Angular step in radians to generate points around great 
            circle.
        """
        for vector in self:
            yield vector.get_great_circle(step)[0]  # because of plot_circles


class PlaneSet(VectorSet):
    """Class that represents a set (collection) of Planes.

    Parameters:
        dcos_data: Is an array of direction cosines.
    """
    item_class = Plane

    def intersection_with(self, other):
        """Returns the intersection of all combinations of 
        planes in this set with the planes in other set as a
        list of lines defined as a VectorSet.

        Parameter:
            other: A PlaneSet like object.
        """
        return self.normalized_cross_with(other).view(LineSet)

    @property
    def attitude(self):
        """Converts this data from direction cosines to attitudes."""
        return sphere_plane(self)


class LineSet(VectorSet):
    """Class that represents a set (collection) of Lines.

    Parameters:
        dcos_data: Is an array of direction cosines.
    """
    item_class = Line

    def planes_with(self, other):
        """Return the list of Planes resulting from the 
        intersection of the combination of all lines in 
        this LineSet with other LineSet like object.

        Parameter:
            other: A LineSet like object.
        """
        return self.normalized_cross_with(other).view(PlaneSet)
