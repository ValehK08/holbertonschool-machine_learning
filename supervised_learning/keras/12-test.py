#!/usr/bin/env python3
"""12-test.py"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """test model"""
    return network.evaluate(data, labels, verbose=verbose)
