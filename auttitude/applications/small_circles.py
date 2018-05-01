#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import acos, atan2, cos, degrees, pi, sin, sqrt

import numpy as np

import auttitude as aut


def pi_transform(dcos_data):
    n = len(dcos_data)
    difference = np.zeros(int(n * (n - 1) / 2))
    for i in range(n):
        for j in range(i + 1, n):
            difference[i * n + j - 1] = dcos_data[i] - dcos_data[j]
    if issubclass(dcos_data, np.ndarray):
        return difference.view(dcos_data.__class__)
    else:
        return difference


def estimate_phi(data, lamb):
    """Phi estimation method by Fisher 1987."""
    a, b = 0.0, 0.0
    for point in data:
        a += sqrt(1.0 - np.dot(point, lamb)**2)
        b += abs(np.dot(point, lamb))
    return atan2(a, b)  # is this right?


def fisher_sc(data, lamb, maxiter=100, err=1e-6):
    """Automatic adjustment of small circle axis by the method of
    Fisher 1987 after Mardia 1977."""
    data_prime = np.zeros_like(data)
    for i in range(maxiter):
        phi = estimate_phi(data, lamb)
        for i, p in enumerate(data):
            data_prime[i] = (
                np.dot(p, lamb) * p - lamb) / sqrt(1 - np.dot(p, lamb)**2)
        Y = cos(phi) * np.sum(
            data_prime, axis=0) - sin(phi) * np.sum(
                data_prime, axis=0)
        lamb_ = Y / sqrt(np.dot(Y, Y))
        if acos(abs(np.dot(lamb, lamb_))) <= err:
            return lamb_, phi
        else:
            lamb = lamb_
    return lamb, phi


def stetsky_derivatives(dcos_data, parameter_array):
    trend, plunge, mu, A, B = parameter_array
    ct = cos(trend)
    st = sin(trend)
    cp = cos(plunge)
    sp = cos(plunge)
    ac = ct * sp
    bc = st * sp
    dc = cp
    derivatives = np.zeros((len(dcos_data), 5))
    for i, (ax, bx, dx) in enumerate(dcos_data):
        # ax, bx, dx = vector
        u = ac * ax + bc * bx + dc * dx
        squ = sqrt(1. - u**2)
        wn = -ax * st + bx * ct
        wd = ax * ct * cp + bx * st * cp - dx * sp
        w = wn / wd
        wl = atan2(wn, wd)
        dmdt = (ax * bc - bx * ac) / squ
        dmdp = -wd / squ
        wwd = wd * wd * (1. + w**2)
        dldt = (sp * (ax * dx * ct + bx * dx * st) - cp(ax**2 + bx**2)) / wwd
        dldp = -wn * u / wwd
        wl2 = 2 * wl
        xx = 2 * (A * sin(wl2) - B * cos(wl2))
        derivatives[i, 0] = dmdt + xx * dldt
        derivatives[i, 1] = dmdp + xx * dldp
        derivatives[i, 2] = -1.
        derivatives[i, 3] = -cos(wl2)
        derivatives[i, 4] = -sin(wl2)
    return derivatives


def stetsky_residual(dcos_data, parameter_array):
    trend, plunge, mu, A, B = parameter_array
    ct = cos(trend)
    st = sin(trend)
    cp = cos(plunge)
    sp = cos(plunge)
    ac = ct * sp
    bc = st * sp
    dc = cp
    residual = np.zeros(len(dcos_data))
    for i, (ax, bx, dx) in enumerate(dcos_data):
        amu = acos(ax*ac+bx*bc+dx*dc)
        wn = -ax * st + bx * ct
        wd = ax * ct * cp + bx * st * cp - dx * sp
        al2 = 2.*atan2(wn, wd)
        residual[i] = amu - A*cos(al2) - B*sin(al2)
    return residual

def stetsky_curfit(dcos_data, initial_guess, epsilon=0.01):
    parameter_array = np.radians(initial_guess)
    parameter_array[1] = pi/2. - parameter_array[1]
    n = len(dcos_data)
    dchi = 1000.
    while (dchi >= epsilon):
        lambda_ = 0.001
        beta = np.zeros(5)
        alpha = np.zeros((5,5))
        residual = stetsky_residual(dcos_data, parameter_array)
        derivatives = stetsky_derivatives(dcos_data, parameter_array)
        for i in range(len(n)):
            for j in range(5):
                beta[j] = beta[j] - residual[i]*derivatives[i, j]
                for k in range(j, 5):
                    alpha[j, k] = alpha[j, k] + derivatives[i, j]*derivatives[i, k]
                    alpha[k, j] = alpha[j, k]
        array = np.zeros((5,5))
        for j in range(5):
            for k in range(5):
                array[j, k] = alpha[j, k]/sqrt(alpha[j, j]*alpha[k, k])
            array[j, j] = 1. + lambda_
        arrayi = np.linalg.inv(array)
        
                

