#!/usr/bin/env python3
""" 2-optimize.py """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ optimizer """

    network.compile(K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    ), loss='categorical_crossentropy', metrics=['Accuracy'])
    return None
