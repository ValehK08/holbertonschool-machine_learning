#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """L2 reg cost"""
    l2_losses = model.losses
    l2_losses_tensor = tf.stack(l2_losses)
    total_cost = cost + l2_losses_tensor

    return total_cost
