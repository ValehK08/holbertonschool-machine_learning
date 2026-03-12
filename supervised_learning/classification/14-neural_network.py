#!/usr/bin/env python3
""" 14-neural_network.py """
import numpy as np


class NeuralNetwork:
    """ Neural Network class """

    def __init__(self, nx, nodes):
        """ initialize """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter """
        return (self.__W1)

    @property
    def b1(self):
        """ getter """
        return (self.__b1)

    @property
    def A1(self):
        """ getter """
        return (self.__A1)

    @property
    def W2(self):
        """ getter """
        return (self.__W2)

    @property
    def b2(self):
        """ getter """
        return (self.__b2)

    @property
    def A2(self):
        """ getter """
        return (self.__A2)

    def forward_prop(self, X):
        """ forward prop """
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """ cost """
        m = Y.shape[1]
        cost = -(np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))))
        cost *= (1 / m)
        return cost

    def evaluate(self, X, Y):
        """ evaluate """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ gradient descent """
        m = Y.shape[1]
        d__W2 = (1 / m) * (np.matmul(A2 - Y, A1.transpose()))
        d__b2 = (1 / m) * (np.sum(A2 - Y, axis=1, keepdims=True))
        dz1 = (np.matmul(self.W2.transpose(), A2 - Y)) * (A1 * (1 - A1))
        d__W1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        d__b1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))
        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)
    
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ train """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return (self.evaluate(X, Y))
