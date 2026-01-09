#!/usr/bin/env python3
""" 13-hierarchy """
import pandas as pd


def analyze(df):
    """
    input -> df
    output -> df with descriptive statistics
    """

    df_desc = df.drop(columns=['Timestamp']).describe()
    return df_desc