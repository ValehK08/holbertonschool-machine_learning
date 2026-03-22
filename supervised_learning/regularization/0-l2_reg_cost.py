#!/usr/bin/env python3
"""0-l2_reg_cost.py"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 regularization cost"""
    norm = 0
    for w, b in weights.items():
        if w[0] == 'W':
            norm = norm + np.linalg.norm(b)
    cost = cost + (lambtha * norm / (2 * m))
    return cost
