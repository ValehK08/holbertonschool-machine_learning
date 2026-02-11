#!/usr/bin/env python3
""" Norm """


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
