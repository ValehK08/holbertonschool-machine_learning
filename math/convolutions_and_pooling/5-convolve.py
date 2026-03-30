#!/usr/bin/env python3
"""5-convolve.py"""
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """convolve it"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (((h - 1) * sh + kh - h) % 2)
        pw = ((w - 1) * sw + kw - w) // 2 + (((w - 1) * sw + kw - w) % 2)
    elif padding == 'valid':
        ph, pw = 0, 0

    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, nc))

    for y in range(out_h):
        for x in range(out_w):
            img_slice = images_padded[:, y*sh:y*sh+kh, x*sw:x*sw+kw, :]
            output[:, y, x, :] = np.tensordot(
                img_slice, kernels, axes=([1, 2, 3], [0, 1, 2])
            )

    return output
