#!/usr/bin/env python3
"""4-moving_average.py"""
import numpy as np


def moving_average(data, beta):
    """Moving Avg"""
    vt = 0
    moving_avg = []
    for t, x in enumerate(data):
        vt = vt * beta + (1 - beta) * x
        bias = 1 - (beta ** (t + 1))
        vt_corrected = vt / bias
        moving_avg.append(vt_corrected)

    return moving_avg
