#!/usr/bin/env python3
""" SIZE ME PLEASE """


def matrix_transpose(matrix):
    """ TRANSPOSE IT """

    return [[matrix[i][k] for i in range(len(matrix))]
            for k in range(len(matrix[0]))]
