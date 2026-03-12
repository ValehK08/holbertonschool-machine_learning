#!/usr/bin/env python3
""" 0-sequential.py """
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model """

    par = [tf.keras.Input(shape=(nx,))]
    for i in range(len(layers)):
        par.append(tf.keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ))
        par.append(tf.keras.layers.Dropout(keep_prob))
    model = tf.keras.Sequential(par)
    return model
