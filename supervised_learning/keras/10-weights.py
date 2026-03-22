#!/usr/bin/env python3
"""10-weights.py"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """save weights"""
    if save_format == 'keras':
        network.save_weights(filename, save_format='keras')


def load_weights(network, filename):
    """load weights"""
    weights = network.load_weights(filename)
