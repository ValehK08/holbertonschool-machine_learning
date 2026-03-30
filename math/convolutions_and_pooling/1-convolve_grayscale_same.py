#!/usr/bin/env python3
"""1-convolve_grayscale_same.py"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolution on greyscale, w/ padding"""
    
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), constant_values=0)
    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            part = padded[:, i:i + kh, j:j + kw] * kernel
            conv[:, i, j] = np.sum(part, axis=(1, 2))
    
    return conv