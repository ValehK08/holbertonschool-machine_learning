#!/usr/bin/env python3
""" SIZE ME PLEASE """
a = [[]]


def matrix_transpose(matrix):
    """ TRANSPOSE IT """

    global a
    for i in range(len(matrix[0])):
        a.append([matrix[k][i] for k in range(len(matrix))])
    return a
