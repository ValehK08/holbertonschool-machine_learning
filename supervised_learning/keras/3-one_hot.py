#!/usr/bin/env python3
""" 3-one_hot.py """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ one hot encoding """

    return K.utils.to_categorical(labels, num_classes=classes)
