#!/usr/bin/env python3
""" 10-index.py """


def index(df):
    """
    input -> df
    output -> df with 'Timestamp' as index
    """

    df = df.set_index('Timestamp')
    return df
