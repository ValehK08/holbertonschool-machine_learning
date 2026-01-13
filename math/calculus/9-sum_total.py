#!/usr/bin/env python3
""" 9-sum_total.py """

def summation_i_squared(n):
    """ Capital Sigma Function """

    if type(n) is not int():
        return None
    return int((n*(n+1)(2*n+1))/6)
