#!/usr/bin/env python3
"""GMM function"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    _, d = X.shape
    pi = np.ones(k) / k
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
