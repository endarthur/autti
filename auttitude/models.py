#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import cos, sin, acos, asin, atan2, degrees, radians, pi
from math import sqrt, log

import numpy as np

# Taken from the pyramid project unmodified, license file on this folder
# reify is a lazy evaluation decorator
from .utils import reify

from .geometry import normalized_cross, sphere_line, sphere_plane
from .grid import default_grid


# Should I really use reify on every property? Won't it clutter memory?
class Vector(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def angle(self, other):
        """Returns the angle (in radians) between both vectors using
        the dot product between them."""
        self_length = self.length
        other_length = sqrt(other.dot(other))
        return acos(self.dot(other)/(self_length*other_length))

    def cross(self, other):
        """Returns the cross product between both vectors."""
        return Vector(np.cross(self, other))

    def normalized_cross(self, other):  # Is this necessary?
        """Returns the normalized cross product between both vectors."""
        return Vector(normalized_cross(self, other))

    @property
    def sphere(self):
        """Returns the spherical coordinates of the normalized vector,
        considering it to be a Line in geological sense, as a
        Trend/Plunge pair in degrees."""
        x, y, z = self/self.length
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
        return direction/direction.length

    @property
    def dip_vector(self):
        """Returns the vector perpendicular to both this vector
        and it's direction vector. If this vector represents a plane,
        the resulting vector represents the maximum slope direction."""
        return Vector(np.cross(self/self.length, self.direction_vector))

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
        return np.array(((0., -self[2], self[1]),
                         (self[2], 0., -self[0]),
                         (-self[1], self[0], 0.)))

    def rotation_matrix(self, theta):
        """Returns the counterclockwise rotation matrix about this vector
        by angle theta."""
        return cos(theta)*np.eye(3) + sin(theta)*self.cross_product_matrix +\
            (1 - cos(theta))*self.projection_matrix

    def great_circle(self, step=1.):
        """Returns an array of n points equally spaced along the great circle
        normal to this vector."""
        theta_range = np.arange(0, 2*pi, radians(step))
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self.direction_vector[:, None]*cos_range
                + self.dip_vector[:, None]*sin_range).T,

    def small_circle(self, alpha, step=1.):
        """Retuns a pair of arrays representing points spaced step along
        both small circles with an semi-apical opening of alpha around
        this vector."""
        sc = self.great_circle(step)[0].T*sin(alpha) + self[:, None]*cos(alpha)
        return sc.T, -sc.T

    def arc(self, other, step=1.):
        """Returns an array of points spaced step along the great circle
        between both vectors."""
        normal = self.rejection_matrix.dot(other)
        normal /= sqrt(normal.dot(normal))
        theta_range = np.arange(0, self.angle(other), radians(step))
        sin_range = np.sin(theta_range)
        cos_range = np.cos(theta_range)
        return (self*cos_range[:, None] + normal*sin_range[:, None]),


class Plane(Vector):
    def intersect(self, other):
        line = Line(self.cross(other))
        line_length = line.length
        return line/line_length if line_length > 0 else line

    @property
    def sphere(self):
        """Returns the spherical coordinates of the plane as a
        Dip Direction/Dip pair, in degrees."""
        x, y, z = self/self.length
        if z > 0:
            x, y = -x, -y
        return degrees(atan2(-x, -y)) % 360, degrees(acos(abs(z)))


class Line(Vector):
    def intersect(self, other):
        plane = Plane(self.cross(other))
        plane_length = plane.length
        return plane/plane_length if plane_length > 0 else plane


class SphericalData(np.ndarray):
    item_class = Vector

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __finalize_array__(self, obj):
        if obj is None:
            return
        if len(self) > 2:
            self.build_statistics()  # maybe defer this until a stat is needed

    def __getitem__(self, x):
        item = super(SphericalData, self).__getitem__(x)
        if np.atleast_2d(item).shape == (1, 3):
            return item.view(self.item_class)
        else:
            return item

    @reify  # Is reify really needed?
    def stats(self):
        return SphericalStatistics(self)

    @property
    def sphere(self):
        return sphere_line(self)

    def count_fisher(self, k=None, grid=None):
        if grid is None:
            grid = default_grid
        return grid.count_fisher(self, k)

    def count_kamb(self, theta=None, grid=None):
        if grid is None:
            grid = default_grid
        return grid.count_kamb(self, theta)

    def intersection(self, other):
        """Returns a SphericalData object containing the normalized cross
        product of all possible pairs betweeen this SphericalData and an
        (n, 3) array-like"""
        vectors = np.zeros((len(self)*len(other), 3))
        i = 0
        for self_vector in self:
            for other_vector in other:
                cross = normalized_cross(self_vector, other_vector)
                vectors[i] = cross if cross[2] < 0 else -cross
                i += 1
        return SphericalData(vectors)

    def angle(self, other):
        """Returns the angles matrix between this Spherical Data and an
        (n, 3) array-like"""
        angles = np.zeros((len(self), len(other)))
        for i, self_vector in enumerate(self):
            for j, other_vector in enumerate(other):
                angles[i, j] = self_vector.angle(other_vector)
        return angles

    def great_circle(self, step=1.):
        """Returns a generator to the great circles of this SphericalData
        vectors."""
        for vector in self:
            yield vector.great_circle(step)[0]  # because of plot_circles


class PlaneData(SphericalData):
    item_class = Plane

    @property
    def sphere(self):
        return sphere_plane(self)


class LineData(SphericalData):
    item_class = Line


class SphericalStatistics(object):
    def __init__(self, data):  # Should this really be built by default?
        self.n = len(data)
        self.resultant_vector = Vector(np.sum(data, axis=0))
        self.mean_resultant_vector = self.resultant_vector/self.n
        self.mean_vector = self.resultant_vector/self.resultant_vector.length
        self.resultant_length = self.resultant_vector.length
        self.mean_resultant_length = self.resultant_length/self.n

        self.resultant_vector_sphere = self.resultant_vector.sphere
        self.fisher_k = (self.n - 1)/(self.n - self.resultant_length)

        direction_tensor = np.dot(np.transpose(data), data)/self.n
        eigenvalues, eigenvectors = np.linalg.eigh(direction_tensor)
        eigenvalues_order = (-eigenvalues).argsort()

        lambda1, lambda2, lambda3 = self.eigenvalues =\
            eigenvalues[eigenvalues_order]
        lambda_sum = eigenvalues.sum()

        self.eigenvectors = [Vector(eigenvector) for eigenvector in
                             eigenvectors[:, eigenvalues_order].T]
        self.eigenvectors_sphere = sphere_line(self.eigenvectors)

        # From Vollmer 1990
        self.vollmer_P = (lambda1 - lambda2)/lambda_sum
        self.vollmer_G = 2*(lambda2 - lambda3)/lambda_sum
        self.vollmer_R = 3*lambda3/lambda_sum

        self.vollmer_classification = ("point", "girdle", "random")[
            np.argmax((self.vollmer_P, self.vollmer_G, self.vollmer_R))]

        self.vollmer_B = self.vollmer_P + self.vollmer_G
        self.vollmer_C = log(lambda1/lambda3)

        # From Woodcock 1977
        self.woodcock_Kx = log(lambda2/lambda3)
        self.woodcock_Ky = log(lambda1/lambda2)
        self.woodcock_C = log(lambda1/lambda3)

        self.woodcock_K = self.woodcock_Ky / self.woodcock_Kx
