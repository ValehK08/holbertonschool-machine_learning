#!/usr/bin/env python3
"""4-convolve_channels.py"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolution on Channels"""
    m, h, w, c = images.shape
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh, sw = stride

    if padding == "same":
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        ph, pw = (0, 0)
    else:
        ph, pw = padding

    h = (h + 2 * ph - kh) // sh + 1
    w = (w + 2 * pw - kw) // sw + 1
    pad = np.pad(images, (
        (0, 0), (ph, ph),
        (pw, pw), (0, 0)
    ))

    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            pad = pad[
                :, i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]
            mul = np.sum(pad * kernel, axis=(1, 2, 3))
            conv[:, i, j] = mul
    return conv
