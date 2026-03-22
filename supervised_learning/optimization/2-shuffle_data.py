#!/usr/bin/env python3
"""2-shuffle_data.py"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle data"""
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X = X[shuffle]
    Y = Y[shuffle]
    return X, Y
