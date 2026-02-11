#!/usr/bin/env python3
""" Binomial """


class Binomial:
    """ Binomial Class """

    def __init__(self, data=None, n=1, p=0.5):
        """ INIT """

        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data) / len(data))
            stdev = 0
            for i in data:
                stdev += (mean - i) ** 2
            stdev = (stdev / len(data)) ** 0.5
            self.p = float(1 - ((stdev ** 2) / mean))
            self.n = int(round(float(mean / self.p)))
            self.p = float(mean / self.n)
    
    def pmf(self, k):
        """ Probability Mass Function """

        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        
        nf = 1
        kf = 1
        n_kf = 1
        for i in range(1, self.n + 1):
            nf *= i
        for i in range(1, k + 1):
            kf *= i
        for i in range(1, self.n - k + 1):
            n_kf *= i
        
        nk = nf / (kf * n_kf)
        return nk * (self.p ** k) * ((1 - self.p) ** (self.n - k))
