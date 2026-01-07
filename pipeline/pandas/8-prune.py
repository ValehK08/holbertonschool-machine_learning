#!/usr/bin/env python3
""" PURIFY """


def prune(df):
    """
    input -> df
    output -> purified df
    """

    df = df.dropna(subset=['Close'], axis=0)
    return df
