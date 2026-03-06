#!/usr/bin/env python3
""" 3-neuron.py """
import numpy as np


class Neuron:
    """ Neuron Class """

    def __init__(self, nx):
        """ Instantiation """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ b getter """
        return self.__b

    @property
    def A(self):
        """ A getter """
        return self.__A

    def forward_prop(self, X):
        """ forward propagation """
        self.__A = (1 / (1 + np.exp(-(np.dot(self.__W, X) + self.__b))))
        return self.__A

    def cost(self, Y, A):
        """ cost function """
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """ evaluate """
        pred = (self.forward_prop(X) >= 0.5).astype(int)
        return pred, self.cost(Y, self.forward_prop(X))
