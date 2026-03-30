#!/usr/bin/env python3
"""3-convolve_grayscale.py"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Convolution on Greyscale"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == "same":
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        ph, pw = (0, 0)
    else:
        ph, pw = padding

    h = (images.shape[1] + 2 * ph - kh) // sh + 1
    w = (images.shape[2] + 2 * pw - kw) // sw + 1
    pad_images = np.pad(images, ((0, 0), (ph, ph),
    (pw, pw)))

    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            pad = pad_images[
                :, i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]
            mul = np.sum(pad * kernel, axis=(1, 2))
            conv[:, i, j] = mul
    return conv