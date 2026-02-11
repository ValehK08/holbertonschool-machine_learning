#!/usr/bin/env python3
""" Norm """


def erf(pi, e, x):
    """ Approximation of erf """

    return (2 / (pi ** 0.5)) * (x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) + ((x ** 9) / 216))
class Normal:
    """ Normal Distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ init """

        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            self.mean = float(sum(data) / len(data))
            s = 0
            for i in data:
                s += (self.mean - i) ** 2
            self.stddev = float(s / len(data)) ** 0.5

    def z_score(self, x):
        """ Z-score """

        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """ x-value """

        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ Probability Distribution Function """

        pi = 3.1415926536
        e = 2.7182818285
        f1 = (1 / (self.stddev * (2 * pi) ** 0.5))
        f2 = (e ** ((-1 / 2) * ((x - self.mean) / (self.stddev)) ** 2))
        return f1 * f2

    def cdf(self, x):
        """ Cumulative Distribution Function """

        pi = 3.1415926536
        e = 2.7182818285

        return (1 / 2) * (1 + erf((x - self.mean)/(self.stddev * (2 ** 0.5))))