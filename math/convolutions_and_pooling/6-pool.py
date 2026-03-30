#!/usr/bin/env python3
"""6-pool.py"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """pool it"""
    m, h, w, c = images.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh, sw = stride

    ph = (h - kh) // sh + 1
    pw = (w - kw) // sw + 1

    pooled = np.zeros((m, ph, pw, c))
    for i in range(ph):
        for j in range(pw):
            pooler = images[
                :, i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]

            if mode == "max":
                pooled[:, i, j] = np.max(pooler, axis=(1, 2))
            if mode == "avg":
                pooled[:, i, j] = np.mean(pooler, axis=(1, 2))

    return pooled
