#!/usr/bin/env python3
""" 12-hierarchy.py """
import pandas as pd

def hierarchy(df1, df2):
    """
    input -> df1, df2
    output -> df with hierarchial index
    """

    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417981]
    df2 = df2.loc[1417411980:1417417981]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.reorder_levels([1, 0], axis=0)
    df = df.sort_index()
    return df
