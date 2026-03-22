#!/usr/bin/env python3
"""11-config.py"""
import tensorflow.keras as K


def save_config(network, filename):
    """save config"""
    model_json = network.to_json()
    with open(filename, "w") as f:
        f.write(model_json)
    return None


def load_config(filename):
    """load config"""
    with open(filename, "r") as f:
        loaded_model_json = f.read()
    model = K.models.model_from_json(loaded_model_json)
    return model
