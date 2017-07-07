#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from .models import LineData


def michael(planes, lines):
    # From Michael (1984)
    # pylint: disable=line-too-long, invalid-name
    A = np.zeros((3*len(planes), 5))
    for i, (n1, n2, n3) in enumerate(planes):
        A[3*i:3*(i+1), :] = ((n1 - n1**3 + n1*n3**2, n2 - 2*n2*n1**2, n3 - 2*n3*n1**2, -n1*n2**2 + n1*n3**2, -2*n1*n2*n3),
                             (-n2*n1**2 + n2*n3**2, n1 - 2*n1*n2**2, -2*n1*n2*n3, n2 - n2**3 + n2*n3**2, n3 - 2*n3*n2**2),
                             (-n3*n1**2 - n3 + n3**3, -2*n1*n2*n3, n1 - 2*n1*n3**2, -n2**2*n3 - n3 + n3**3, n2 - 2*n2*n3**2))
    stresses, residuals = np.linalg.lstsq(A, lines.ravel())[:2]
    s11, s12, s13, s22, s23 = -stresses
    stress_matrix = np.array(((s11, s12, s13),
                              (s12, s22, s23),
                              (s13, s23, -(s11 + s22))))
    return stress_matrix, residuals.view(np.ndarray)


def principal_stresses(stress_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(stress_matrix)
    eigenvalues_order = (eigenvalues).argsort()
    eigenvectors = LineData(eigenvectors[:, eigenvalues_order].T)
    return eigenvectors, eigenvalues
