#!/usr/bin/env python3
"""9-Adam.py"""
import numpy as np


def update_variables_Adam(
    alpha, beta1, beta2, epsilon,
    var, grad, v, s, t
):
    """Lilith hates Adam"""
    vt = beta1 * v + (1 - beta1) * grad
    st = beta2 * s + (1 - beta2) * grad ** 2
    vt_hat = vt / (1 - beta1 ** t)
    st_hat = st / (1 - beta2 ** t)
    var = var - alpha * vt_hat / (np.sqrt(st_hat) + epsilon)
    return var, vt, st
