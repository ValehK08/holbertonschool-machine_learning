#!/usr/bin/env python3
""" 3-specificity.py """
import numpy as np


def specificity(confusion):
    """ SPECIFICITY """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - TP - FP - FN

    spec = TN / (TN + FP)
    return spec
