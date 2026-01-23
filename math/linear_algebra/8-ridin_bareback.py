#!/usr/bin/env python3
""" RIDIN' BAREBACK """


def mat_mul(mat1, mat2):
    """ matrix multiplication """
    a = []
    for i in list(map(
            lambda x: [
                list(
                    map(lambda x: list(x), zip(*([x]+[i])))
                ) for i in list(
                    map(
                        lambda x: list(x), list(zip(*mat2))
                    )
                )
            ], mat1)):
        b = []
        for j in i:
            s = 0
            for k in j:
                p = 1
                for z in k:
                    p *= z
                s += p
            b.append(s)
        a.append(b)
    return a
