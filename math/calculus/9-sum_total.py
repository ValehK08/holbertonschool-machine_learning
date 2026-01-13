#!/usr/bin/env python3
""" 9-sum_total.py """


def summation_i_squared(n):
    """ Capital Sigma Function """

    if (not isinstance(n, int)) and n < 1:
        return None
    return int((n*(n+1)*(2*n+1))/6)
