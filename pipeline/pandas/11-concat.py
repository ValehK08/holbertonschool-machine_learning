#!/usr/bin/env python3
""" 11-concat.py """
import pandas as pd


def concat(df1, df2):
    """
    input -> df1, df2
    output -> concatenated df1 and df2 into df
    """

    index = __import__('10-index').index
    df1 = df1.index('Timestamp')
    df2 = df2.index('Timestamp')
    df2 = df2.loc[:1417411921]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return df
