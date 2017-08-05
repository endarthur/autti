#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from math import asin, cos, degrees, log, pi, radians, sin, sinh

import numpy as np

from auttitude import datamodels
from auttitude.io import dcos_line, sphere_line


# maybe add option to build grid from projecting regular
# plane grid to sphere
class SphericalGrid(object):
    def __init__(self, node_spacing=2.5):
        """Creates a hemi-spherical counting grid by tesselation"""
        nodes = [(0., 90.)]
        spacing = radians(node_spacing)
        for phi in np.arange(node_spacing, 90., node_spacing):
            azimuth_spacing = degrees(2 * asin(
                (sin(spacing / 2) / sin(radians(phi)))))
            for theta in np.arange(0., 360., azimuth_spacing):
                nodes.append((theta + phi + node_spacing / 2, 90. - phi))
                nodes.append((theta + phi + node_spacing / 2, phi - 90.))
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


class SphericalStatistics(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, data):  # Should this really be built by default?
        n = len(data)
        self.resultant_vector = datamodels.Vector(np.sum(data, axis=0))
        self.mean_resultant_vector = self.resultant_vector / n
        self.mean_vector = self.resultant_vector / self.resultant_vector.length
        self.resultant_length = self.resultant_vector.length
        self.mean_resultant_length = self.resultant_length / n

        self.resultant_vector_attitude = self.resultant_vector.to_attitude
        self.fisher_k = (n - 1) / (n - self.resultant_length)

        direction_tensor = np.dot(np.transpose(data), data) / n
        eigenvalues, eigenvectors = np.linalg.eigh(direction_tensor)
        eigenvalues_order = (-eigenvalues).argsort()

        lambda1, lambda2, lambda3 = self.eigenvalues =\
            eigenvalues[eigenvalues_order]
        lambda_sum = eigenvalues.sum()

        self.eigenvectors = [
            datamodels.Vector(eigenvector)
            for eigenvector in eigenvectors[:, eigenvalues_order].T
        ]
        self.eigenvectors_attitude = sphere_line(self.eigenvectors)

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
    mean_vector = datamodels.Vector(mean_vector)
    direction_vector = mean_vector.direction_vector
    dip_vector = mean_vector.dip_vector
    kappa = kappa
    theta_sample = np.random.uniform(0, 2 * pi, n)
    alpha_sample = np.random.vonmises(0, kappa / 2., n)  # Why?
    return datamodels.VectorSet(
        ((direction_vector[:, None] * np.cos(theta_sample) +
          dip_vector[:, None] * np.sin(theta_sample)) * np.sin(alpha_sample) +
         mean_vector[:, None] * np.cos(alpha_sample)).T)


def sample_uniform(n):
    """Sample n vectors for the uniform distribution on the sphere."""
    samples = np.random.normal(size=(n, 3))
    return datamodels.VectorSet(
        samples / np.linalg.norm(samples, axis=1)[:, None])


DEFAULT_GRID = SphericalGrid(node_spacing=2.5)
