#!/usr/bin/env python3
"""updates a Gaussian Process"""
import numpy as np


class GaussianProcess:
    """A class that does the trick"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the class"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Compute the kernel between two points"""
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """Predict using the GP model"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        cov_s = cov_s.diagonal()

        return mu_s, cov_s

    def update(self, X_new, Y_new):
        """Update the GP model"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
