#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from auttitude.datamodels import LineSet
from auttitude.stats import DEFAULT_GRID


def michael(planes, lines):
    """Linear paleostress inversion by Michael (1984) method.

    Estimate paleostress from fault data using the method detailed in
    Michael (1984). Needs at least two faults (plane/line pairs) to work.

    Args:
        planes: (M, 3) array_like
                An array-like containing the direction cosines of the fault
                planes.
        lines: (M, 3) array_like
               An array-like containing the direction consines of the fault
               lines, respective to each plane.

    Returns:
        stress_matrix: (3, 3) ndarray
                       Least-squares solution to the inversion method.
        residual: float
                  Sum of residuals for the solution.

    Notes
    -----
    To linearize the inversion problem, _[1] assumes a priori that the
    tangential traction magnitude is constant on each fault plane, and as only
    relative stresses may be obtained by inversion, sets this traction to 1.
    This allows for a least squares solution to the problem.


    References
    ----------
    .. [1] Michael, A. J. (1984). Determination of stress from slip data:
           faults and folds. Journal of Geophysical Research: Solid Earth,
           89(B13), 11517-11526.
    """
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
    return stress_matrix, residuals[0]


def angelier_graphical(planes, lines, grid=DEFAULT_GRID.grid):
    try:  # Try calculating directly with numpy arrays
        result = (grid.dot(np.transpose(planes))*grid.dot(np.transpose(lines))).sum(axis=1)*2
        return result
    except MemoryError:
        result = np.zeros((grid.shape[0], 1))
        for input_node, output_node in zip(grid, result):
            output_node[:] = 2*input_node.dot(planes)*input_node.dot(lines).sum()
    return result


def principal_stresses(stress_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(stress_matrix)
    eigenvalues_order = (eigenvalues).argsort()
    eigenvectors = LineSet(eigenvectors[:, eigenvalues_order].T)
    return eigenvectors, eigenvalues
