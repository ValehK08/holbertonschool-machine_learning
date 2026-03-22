#!/usr/bin/env python3
"""6-momentum.py"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Create optimizer"""
    optimizer = tf.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
