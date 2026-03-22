#!/usr/bin/env python3
"""0-norm_constants.py"""

import numpy as np


def normalization_constants(X):
    """Normalization constants"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
