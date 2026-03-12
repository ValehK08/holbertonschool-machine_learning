#!/usr/bin/env python3
""" 1-input.py """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model """

    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
