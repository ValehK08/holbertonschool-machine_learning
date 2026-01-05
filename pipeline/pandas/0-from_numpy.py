#!/usr/bin/env python3
""" FROM NUMPY """

import pandas as pd
def from_numpy(array):
    """FROM NUMPY ARRAY TO DATAFRAME"""

    cols = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    df = pd.DataFrame(data=array, columns=cols[:array.shape[1]])
    return df