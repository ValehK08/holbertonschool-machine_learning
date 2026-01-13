#!/usr/bin/env python3
""" 17-integrate.py """


def poly_integral(poly, C=0):
    """ polynomial integration """

    if not isinstance(poly, list) or len(poly) == 0:
        return None
    elif not isinstance(C, (int, float)):
        return None
    else:
        poly = [poly[x]/(x+1) for x in range(len(poly))]
        for k in range(len(poly)):
            if int(poly[k]) == poly[k]:
                poly[k] = int(k)
        poly.insert(0, C)
        return poly
