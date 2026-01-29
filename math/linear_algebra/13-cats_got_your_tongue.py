#!/usr/bin/env python3
""" catsgotyourtongue """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """NPCAT"""

    return np.concatenate((mat1, mat2), axis=axis)
