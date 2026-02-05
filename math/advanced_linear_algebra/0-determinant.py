#!/usr/bin/env python3
""" DETERMINATE """


def determinant(matrix):
    """ Find determinant of matrix """

    if matrix == [[]]:
        return 1
    
    if matrix == []:
        return 0

    if not isinstance(matrix[0], list):
        raise TypeError('matrix must be a list of lists') 
    
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    
    if len(matrix) == 1:
        det = matrix[0][0]
    
    if len(matrix[0]) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    if len(matrix[0]) == 3:
        det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[2][0] * matrix[1][2]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])

    return det
