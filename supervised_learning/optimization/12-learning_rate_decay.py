#!/usr/bin/env python3
"""12-learning_rate_decay.py"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Learning rate decay"""
    alpha = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return alpha
