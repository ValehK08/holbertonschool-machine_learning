#!/usr/bin/env python3
"""1-normalize.py"""

import numpy as np


def normalize(X, m, s):
    """Normalize it"""
    return (X - m) / s
