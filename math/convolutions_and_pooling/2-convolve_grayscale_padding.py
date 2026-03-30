#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolution on Greyscale, w/ padding"""
    kh, kw = kernel.shape[0], kernel.shape[1]
    images = np.pad(images, (
        (0, 0), (padding[0], padding[0]), (padding[1], padding[1])
    ), constant_values=0)
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    conv = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            part = images[:, i:i + kh, j:j + kw] * kernel
            conv[:, i, j] = np.sum(part, axis=(1, 2))
    return conv
