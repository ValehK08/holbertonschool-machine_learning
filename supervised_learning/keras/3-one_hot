#!/usr/bin/env python3
""" 3-one_hot.py """
import numpy as np

def one_hot(labels, classes=None):
    """ one hot encoding """
    print(labels.shape[0], len(set(labels)))
    one_hot_matrix = np.zeros((labels.shape[0], np.max(labels) + 1))

    for i in range(labels.shape[0]):
        one_hot_matrix[i, labels[i]] = 1

    return one_hot_matrix
