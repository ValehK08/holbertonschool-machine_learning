#!/usr/bin/env python3
"""dataframe -> array"""


def array(df):
    """
    input -> dataframe
    output -> array
    """

    df = df[['High', 'Close']].tail(10)
    arr = df.to_numpy()
    return arr
