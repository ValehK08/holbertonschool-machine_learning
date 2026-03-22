#!/usr/bin/env python3
"""5-momentum.py"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update the momentum"""
    vt = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vt
    return w, vt
