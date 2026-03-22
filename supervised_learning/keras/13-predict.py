#!/usr/bin/env python3
"""13-predict.py"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predict"""
    return network.predict(data, verbose=verbose)
