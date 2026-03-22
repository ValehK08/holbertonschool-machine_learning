#!/usr/bin/env python3
"""3-mini_batch.py"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create batches"""
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[0]

    number_of_batches = m // batch_size
    for i in range(number_of_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size

        X_batch = X_shuffled[start_index:end_index]
        Y_batch = Y_shuffled[start_index:end_index]

        mini_batches.append([X_batch, Y_batch])

    if m % batch_size != 0:
        X_batch = X_shuffled[number_of_batches * batch_size:]
        Y_batch = Y_shuffled[number_of_batches * batch_size:]
        mini_batches.append([X_batch, Y_batch])

    return mini_batches
