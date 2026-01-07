#!/usr/bin/env python3
"""Sort by High"""


def high(df):
    """
    input -> df
    output -> df sorted by High
    """

    df = df.sort_values('High', ascending=False)
    return df
