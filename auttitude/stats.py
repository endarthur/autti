#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from math import asin, cos, degrees, log, pi, radians, sin, sinh, sqrt

import numpy as np

import auttitude as at
from auttitude.io import dcos_line, sphere_line


# maybe add option to build grid from projecting regular
# plane grid to sphere
class SphericalGrid(object):
    """
    This class represents an quasi-regular spherical grid with a given distance
    between nodes.

    Creates a hemi-spherical counting grid by tesselation.

        Parameters:
            node_spacing: Distance between nodes in degrees.

    """
    def __init__(self, node_spacing=2.5):
        nodes = [(0., 90.)]
        spacing = radians(node_spacing)
        for phi in np.arange(node_spacing, 90., node_spacing):
            azimuth_spacing = degrees(2 * asin(
                (sin(spacing / 2) / sin(radians(phi)))))
            for theta in np.linspace(0., 360. - azimuth_spacing, int(360. // azimuth_spacing)):
                nodes.append((theta + phi + node_spacing / 2, 90. - phi))
                nodes.append((theta - 180 + phi + node_spacing / 2, phi - 90.))
        for theta in np.arange(0., 360., node_spacing):
            nodes.append(((theta + 90. + node_spacing / 2) % 360., 0.))
        self.node_attitudes = nodes
        self.grid = dcos_line(np.array(nodes))

    @staticmethod
    def optimize_k(data):
        """Optimizes the value of K from the data, using Diggle and Fisher (88)
            method."""
        from scipy.optimize import minimize_scalar

        def obj(k):  # objective function to be minimized
            W = np.exp(k*(np.abs(np.dot(data, np.transpose(data)))))\
                * (k/(4*pi*sinh(k+1e-9)))
            np.fill_diagonal(W, 0.)
            return -np.log(W.sum(axis=0)).sum()

        return minimize_scalar(obj).x

    def count_fisher(self, data, k=None):
        """Performs axial data counting as in Robin and Jowett (1986).
        Will estimate an appropriate k if not given."""
        if k is None:
            k = self.optimize_k(data)  # Its a better estimate than R&J 86.
        return self.count(
            data,
            lambda nodes, data, k: np.exp(
                k * (np.abs(np.dot(nodes, np.transpose(data))) - 1)),
            k)

    def count_kamb(self, data, theta=None):
        """Performs data counting as in Robin and Jowett (1986) based on
        Kamb (1956). Will estimate an appropriate counting angle theta
        if not give."""
        if theta is None:
            theta = (len(data) - 1.0) / (len(data) + 1.0)
        else:
            theta = cos(radians(theta))
        return self.count(
            data,
            lambda nodes, data, theta: np.where(
                np.abs(np.dot(nodes, np.transpose(data))) >= theta, 1, 0),
            theta)

    def count(self, data, function, *args, **kwargs):
        """Generic counting grid method that accepts a function which
        receives the grid (or a node in the grid), the data and the additional
        arguments and keyword arguments passed to this method."""
        try:  # Try calculating directly with numpy arrays
            return function(self.grid, data, *args, **kwargs).sum(axis=1)
        except (MemoryError, ValueError):
            result = np.zeros(self.grid.shape[0])
            for i, input_node in enumerate(self.grid):
                result[i] = function(input_node, data, *args, **kwargs).sum()
        return result


class CircularGrid(object):
    def __init__(self, spacing=1., offset=0., **kwargs):
        self.spacing = spacing
        self.grid = self.build_grid(spacing, offset)

    def build_grid(self, spacing, offset=0., from_=0., to_=2 * pi):
        s = radians(spacing)
        o = radians(offset)
        theta_range = np.arange(o, 2 * pi + o, s)
        theta_range = theta_range[np.logical_and(theta_range >= from_,
                                                 theta_range <= to_)]
        return np.array((np.sin(theta_range), np.cos(theta_range))).T

    def cdis(self, data, nodes=None, axial=False):
        nodes = self.grid if nodes is None else nodes
        d = np.clip(
            np.dot(nodes, np.transpose(data)) / np.linalg.norm(data, axis=1),
            -1, 1)
        if axial:
            d = np.abs(d)
        return d

    def count(self,
              data,
              aperture=None,
              axial=False,
              spacing=None,
              offset=0,
              nodes=None,
              data_weight=None):
        aperture = radians(aperture) / 2. if aperture is not None else radians(
            self.spacing) / 2.
        if nodes is None:
            nodes = self.grid if spacing is None else self.build_grid(
                spacing, offset)
        spacing = radians(
            self.spacing) / 2 if spacing is None else radians(spacing) / 2
        c = cos(aperture)
        n = data.shape[0]
        data_weight = np.ones(n) if data_weight is None else data_weight
        return np.where(self.cdis(data, nodes, axial=axial) >= c,\
                        data_weight, 0.).sum(axis=1)[:,None]/data_weight.sum()

    def count_munro(self,
                    data,
                    weight=.9,
                    aperture=10.,
                    axial=False,
                    spacing=None,
                    offset=0,
                    nodes=None,
                    data_weight=None):
        spacing = 1 if spacing is None else spacing
        if nodes is None:
            nodes = self.grid if spacing is None else self.build_grid(
                spacing, offset)
        d = self.cdis(data, nodes, axial=axial)
        aperture = radians(aperture) / 2. if aperture is not None else radians(
            self.spacing) / 2.
        c = cos(aperture)
        theta = np.arccos(d) * pi / aperture
        data_weight = np.ones(
            data.shape[0]) if data_weight is None else data_weight
        upscale = 1. + 2. * np.power(
            weight, np.arange(0., aperture, radians(spacing))).sum()
        return (np.where(d >= c, data_weight, 0) * np.power(weight, theta)
                ).sum(axis=1)[:, None] * upscale / data_weight.sum()


class CircularStatistics(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, data):  # Should this really be built by default?
        n = len(data)
        self.resultant_vector = at.datamodels.Vector(np.sum(data, axis=0))
        self.mean_resultant_vector = self.resultant_vector / n
        self.mean_vector = self.resultant_vector / self.resultant_vector.length
        self.resultant_length = self.resultant_vector.length
        self.mean_resultant_length = self.resultant_length / n

        self.resultant_vector_attitude = self.resultant_vector.attitude

        self.circular_variance = 1 - self.mean_resultant_length
        self.circular_standard_deviation = sqrt(
            -2 * log(1 - self.circular_variance))
        # self.circular_mean_direction_axial, self.circular_confidence_axial =\
        #     self.estimate_circular_confidence(axial=True)
        # self.circular_mean_direction, self.circular_confidence =\
        #     self.estimate_circular_confidence(axial=False)

        self.fisher_k = (n - 1) / (n - self.resultant_length)

        direction_tensor = np.dot(np.transpose(data), data) / n
        eigenvalues, eigenvectors = np.linalg.eigh(direction_tensor)
        eigenvalues_order = (-eigenvalues).argsort()

        self.eigenvalues = eigenvalues[eigenvalues_order]

        self.eigenvectors = [
            at.datamodels.Vector(eigenvector)
            for eigenvector in eigenvectors[:, eigenvalues_order].T
        ]
        self.eigenvectors_attitude = sphere_line(self.eigenvectors)


class SphericalStatistics(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, data):  # Should this really be built by default?
        n = len(data)
        self.resultant_vector = at.datamodels.Vector(np.sum(data, axis=0))
        self.mean_resultant_vector = self.resultant_vector / n
        self.mean_vector = self.resultant_vector / self.resultant_vector.length
        self.resultant_length = self.resultant_vector.length
        self.mean_resultant_length = self.resultant_length / n

        self.resultant_vector_attitude = self.resultant_vector.attitude
        self.fisher_k = (n - 1) / (n - self.resultant_length)

        direction_tensor = np.dot(np.transpose(data), data) / n
        eigenvalues, eigenvectors = np.linalg.eigh(direction_tensor)
        eigenvalues_order = (-eigenvalues).argsort()

        lambda1, lambda2, lambda3 = self.eigenvalues =\
            eigenvalues[eigenvalues_order]
        lambda_sum = eigenvalues.sum()

        self.eigenvectors = [
            at.datamodels.Vector(eigenvector)
            for eigenvector in eigenvectors[:, eigenvalues_order].T
        ]
        self.eigenvectors_attitude = sphere_line(self.eigenvectors)

        # Check for divide by zero on stats?
        # From Vollmer 1990
        self.vollmer_P = (lambda1 - lambda2) / lambda_sum
        self.vollmer_G = 2 * (lambda2 - lambda3) / lambda_sum
        self.vollmer_R = 3 * lambda3 / lambda_sum

        self.vollmer_classification = ("point", "girdle", "random")[np.argmax((
            self.vollmer_P, self.vollmer_G, self.vollmer_R))]

        self.vollmer_B = self.vollmer_P + self.vollmer_G
        self.vollmer_C = log(lambda1 / lambda3)

        # From Woodcock 1977
        self.woodcock_Kx = log(lambda2 / lambda3)
        self.woodcock_Ky = log(lambda1 / lambda2)
        self.woodcock_C = log(lambda1 / lambda3)

        self.woodcock_K = self.woodcock_Ky / self.woodcock_Kx


def sample_fisher(mean_vector, kappa, n):
    """Samples n vectors from von Mises-Fisher distribution."""
    mean_vector = at.datamodels.Vector(mean_vector)
    direction_vector = mean_vector.direction_vector
    dip_vector = mean_vector.dip_vector
    kappa = kappa
    theta_sample = np.random.uniform(0, 2 * pi, n)
    alpha_sample = np.random.vonmises(0, kappa / 2., n)  # Why?
    return at.datamodels.VectorSet(
        ((direction_vector[:, None] * np.cos(theta_sample) +
          dip_vector[:, None] * np.sin(theta_sample)) * np.sin(alpha_sample) +
         mean_vector[:, None] * np.cos(alpha_sample)).T)


def sample_uniform(n):
    """Sample n vectors for the uniform distribution on the sphere."""
    samples = np.random.normal(size=(n, 3))
    return at.datamodels.VectorSet(
        samples / np.linalg.norm(samples, axis=1)[:, None])


DEFAULT_GRID = SphericalGrid(node_spacing=2.5)
