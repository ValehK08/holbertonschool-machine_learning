#!/usr/bin/env python3
"""13-batch_norm.py"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Batch normalization."""
    myu = np.mean(Z, axis=0)
    sigma_2 = np.var(Z, axis=0)
    z_norm = (Z - myu) / (np.sqrt(sigma_2 + epsilon))
    return z_norm * gamma + beta
