#!/usr/bin/env python3
""" SIZE ME PLEASE """
a = []


def matrix_shape(matrix):
    """ Return matrix shape """
    global a
    a.append(len(matrix))
    if isinstance(matrix[0], list):
        matrix_shape(matrix[0])
    return a
