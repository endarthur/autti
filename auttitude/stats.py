#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import log

import numpy as np

import models
from geometry import sphere_line


class SphericalStatistics(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, data):  # Should this really be built by default?
        self.n = len(data)
        self.resultant_vector = models.Vector(np.sum(data, axis=0))
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

        self.eigenvectors = [models.Vector(eigenvector) for eigenvector in
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
