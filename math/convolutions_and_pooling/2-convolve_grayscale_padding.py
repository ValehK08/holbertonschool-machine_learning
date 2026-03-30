#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolution on Greyscale, w/ padding"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    images = np.pad(images, (
        (0, 0), (padding[0], padding[0]), (padding[1], padding[1])
    ), constant_values=0)
    conv = np.zeros((m, images.shape[1] - kh + 1, images.shape[2] - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            part = images[:, i:i + kh, j:j + kw] * kernel
            conv[:, i, j] = np.sum(part, axis=(1, 2))
    return conv
