#!/usr/bin/env python3
""" 10-matisse.py """


def poly_derivative(poly):
    """ polynomial derivative """

    if not isinstance(poly, list) or len(poly) == 0:
        return None
    elif all(x == 0 for x in poly[1::]):
        return [0]
    else:
        poly = [k*poly[k] for k in range(1, len(poly))]
        return poly