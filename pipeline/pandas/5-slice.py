#!/usr/bin/env python3
"""slice the dataframe"""


def slice(df):
    """
    input -> df
    output -> sliced df
    """

    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_BTC']]
    return df
