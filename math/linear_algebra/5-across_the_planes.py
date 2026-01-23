#!/usr/bin/env python3
""" ACROSS THE PLANES """


def add_matrices2D(mat1, mat2):
    """ add matrices """
    if len(mat1) != len(mat2):
        return None
    a = []
    for i in range(len(mat1)):
        a.append([
            sum(k)
            for k in list(zip(*[(mat1+mat2)[i]]+[(mat1+mat2)[i+len(mat1)]]))
            ])
    return a
