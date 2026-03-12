#!/usr/bin/env python3
""" 24-one_hot_encode.py """
import numpy as np


def one_hot_encode(Y, classes):
    """ one_hot_encode"""
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.zeros((classes, m))
        for i in range(Y.shape[0]):
            one_hot[Y[i], i] = 1
        return one_hot
    except Exception as err:
        return None
