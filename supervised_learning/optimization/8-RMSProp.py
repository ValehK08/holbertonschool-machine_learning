#!/usr/bin/env python3
"""8-RMSProp.py"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Creates a RMSProp operation"""
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2, epsilon=epsilon
    )
    return optimizer
