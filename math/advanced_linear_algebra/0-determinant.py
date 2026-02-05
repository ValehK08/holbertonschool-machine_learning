#!/usr/bin/env python3
""" DETERMINATE """


def determinant(matrix):
    """ Find determinant """

    if any(not isinstance(i, list) for i in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(len(matrix)):
        minor = [k[:i] + k[i + 1:] for k in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(minor)

    return det
