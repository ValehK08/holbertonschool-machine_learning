#!/usr/bin/env python3
""" 2-precision.py """
import numpy as np


def precision(confusion):
    """ PRECISION """

    prec = np.zeros((confusion.shape[0],))
    for i in range(confusion.shape[0]):
        prec[i] = (confusion.T[i, i]/np.sum(confusion.T[i]))
    return prec
