#!/usr/bin/env python3
"""6-dropout_create_layer.py"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a dropout layer"""
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    x = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)

    x = tf.keras.layers.Dropout(rate=1 - keep_prob)(
        x, training=training
    )

    return x
