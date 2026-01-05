#!/usr/bin/env python3
"""slice the dataframe"""

import pandas as pd


def slice(df):
    """
    input -> df
    output -> sliced df
    """

    df = df[['High', 'Low', 'Close', 'Volume_BTC']]
    return df.loc[61, :]
