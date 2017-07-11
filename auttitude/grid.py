#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import cos, sin, asin, degrees, radians, pi
from math import sinh

import numpy as np

from geometry import dcos_line


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


# maybe add option to build grid from projecting regular
# plane grid to sphere
class SphericalGrid(object):
    def __init__(self, node_spacing=2.5):
        """Creates a hemi-spherical counting grid by tesselation"""
        nodes = [(0., 90.)]
        spacing = radians(node_spacing)
        for phi in np.arange(node_spacing, 90., node_spacing):
            azimuth_spacing = degrees(2*asin((sin(spacing/2)
                                      / sin(radians(phi)))))
            for theta in np.arange(0., 360., azimuth_spacing):
                nodes.append((theta+phi + node_spacing/2, 90. - phi))
                nodes.append((theta+phi + node_spacing/2, phi - 90.))
        for theta in np.arange(0., 360., node_spacing):
            nodes.append(((theta + 90. + node_spacing/2) % 360., 0.))
        self.node_attitudes = nodes
        self.grid = dcos_line(np.array(nodes))


    def count_fisher(self, data, k=None):
        """Performs axial data counting as in Robin and Jowett (1986).
        Will estimate an appropriate k if not given."""
        if k is None:
            k = optimize_k(data)  # Its a better estimate than R&J 86.
        return self.count(data, fisher_distance, k)

    def count_kamb(self, data, theta=None):
        """Performs data counting as in Robin and Jowett (1986) based on
        Kamb (1956). Will estimate an appropriate counting angle theta
        if not give."""
        if theta is None:
            theta = (len(data)-1.0)/(len(data)+1.0)
        else:
            theta = cos(radians(theta))
        return self.count(data, angle_treshold, theta)

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

def fisher_distance(nodes, data, k):
    return np.exp(k * (np.abs(np.dot(nodes, np.transpose(data))) - 1))

def angle_treshold(nodes, data, theta):
    return np.where(np.abs(np.dot(nodes, np.transpose(data))) >= theta, 1, 0)


default_grid = SphericalGrid(node_spacing=2.5)
