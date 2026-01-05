#!/usr/bin/env python3
"""slice the dataframe"""


def slice(df):
    """
    input -> df
    output -> sliced df
    """

    df = df.loc[:, ['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60]
    return df
