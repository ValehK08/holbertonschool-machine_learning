#!/usr/bin/env python3
"""0-convolve_grayscale_valid.py"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Convolution on Greyscale"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    conv = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            part = images[:, i:i + kh, j:j + kw] * kernel
            conv[:, i, j] = np.sum(part, axis=(1, 2))
    return conv_imgs
