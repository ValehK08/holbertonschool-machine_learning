#!/usr/bin/env python3
""" 2-identity_block.py """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """I-block"""

    F11, F3, F12 = filters
    he_init = K.initializers.he_normal(seed=0)

    X_shortcut = A_prev

    X = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1), strides=(s, s),
        padding='valid', kernel_initializer=he_init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), strides=(1, 1),
        padding='same', kernel_initializer=he_init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), strides=(1, 1),
        padding='valid', kernel_initializer=he_init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X_shortcut = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), strides=(s, s),
        padding='valid', kernel_initializer=he_init
    )(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
