#!/usr/bin/env python3
""" GETTIN' COZY """


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat the matrices """
    if axis == 0:
        return mat1+mat2
    else:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i]+mat2[i] for i in range(len(mat1))]
