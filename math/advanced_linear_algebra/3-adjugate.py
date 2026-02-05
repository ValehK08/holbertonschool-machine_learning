#!/usr/bin/env python3
""" ADJUGATE """


def adjugate(matrix):
    """ ADJOINT """

    m = cofactor(matrix)

    return list(map(list, zip(*m)))


def cofactor(matrix):
    """ COFACTORIZE """

    if any(not isinstance(i, list) for i in matrix) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    if any(len(i) != len(matrix) for i in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    m = minor(matrix)
    for i in range(len(m)):
        for k in range(len(m[i])):
            m[i][k] *= (-1) ** (i + k)
    return m


def minor(matrix):
    """ FIND THE MINOR """

    if any(not isinstance(i, list) for i in matrix) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    if any(len(i) != len(matrix) for i in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    m = [[] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            m[i].append(determinant([
                row[:j] + row[j+1:]
                for row in (matrix[:i]+matrix[i+1:])
                ]))
    return m


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
