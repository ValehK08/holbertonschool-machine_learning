#!/usr/bin/env python3
""" 1-sensitivity.py """
import numpy as np


def sensitivity(confusion):
    """ SENSITIVITY """

    sense = np.zeros((confusion.shape[0],))
    for i in range(confusion.shape[0]):
        sense[i] = (confusion[i, i]/np.sum(confusion[i]))
    return sense
