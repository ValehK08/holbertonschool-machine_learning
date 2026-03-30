#!/usr/bin/env python3
"""5-convolve.py"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """convolve"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        ph, pw = (0, 0)
    else:
        ph, pw = padding

    h_n = (h + 2 * ph - kh) // sh + 1
    w_n = (w + 2 * pw - kw) // sw + 1
    pad = np.pad(images, (
        (0, 0), (ph, ph),
        (pw, pw), (0, 0)
    ))

    conv = np.zeros((h_n, h, w, nc))
    for i in range(h):
        for j in range(w):
            for k in range(nc):
                pad = pad[
                    :, i * sh:i * sh + kh,
                    j * sw:j * sw + kw
                ]
                mul = np.sum(
                    pad * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )
                conv[:, i, j, k] = mul
    return conv