#!/usr/bin/env python3
""" 25-one_hot_decode.py """
import numpy as np


def one_hot_decode(one_hot):
    """ one hot decode """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return one_hot.transpose().argmax(axis=1)
