#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import pi

import numpy as np

import models


def fisher_sample(mean_vector, kappa, n):
    """Samples n vectors from von Mises-Fisher distribution."""
    mean_vector = models.Vector(mean_vector)
    direction_vector = mean_vector.direction_vector
    dip_vector = mean_vector.dip_vector
    kappa = kappa
    theta_sample = np.random.uniform(0, 2*pi, n)
    alpha_sample = np.random.vonmises(0, kappa/2., n)  # Why?
    return models.SphericalData(((direction_vector[:, None]*np.cos(theta_sample)
                                  + dip_vector[:, None]*np.sin(theta_sample))
                                 *np.sin(alpha_sample)
                                 + mean_vector[:, None]*np.cos(alpha_sample)).T)

def uniform_sample(n):
    """Sample n vectors for the uniform distribution on the sphere."""
    samples = np.random.normal(size=(n, 3))
    return models.SphericalData(samples/np.linalg.norm(samples, axis=1)[:, None])
