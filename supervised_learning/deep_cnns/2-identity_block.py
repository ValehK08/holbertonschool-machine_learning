#!/usr/bin/env python3
""" 2-identity_block.py """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """I-block"""

    F11, F3, F12 = filters
    he_init = initializers.he_normal(seed=0)

    X_shortcut = A_prev

    X = layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', kernel_initializer=he_init)(A_prev)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', kernel_initializer=he_init)(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                      padding='valid', kernel_initializer=he_init)(X)
    X = layers.BatchNormalization(axis=3)(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X