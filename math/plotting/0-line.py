#!/usr/bin/env python3
""" 0-line.py """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """ line graph """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='r')
    plt.axis([0, 10, None, None])
    plt.show()
