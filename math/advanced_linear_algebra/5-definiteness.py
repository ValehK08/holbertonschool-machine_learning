#!/usr/bin/env python3
""" DEFINE """
import numpy as np


def definiteness(matrix):
    """ DEFINITENESS """

    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) < 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    vals = np.linalg.eigvals(matrix)

    if np.all(vals > 0):
        return "Positive definite"
    elif np.all(vals < 0):
        return "Negative definite"
    elif np.all(vals >= 0) and np.any(vals == 0):
        return "Positive semi-definite"
    elif np.all(vals <= 0) and np.any(vals == 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
