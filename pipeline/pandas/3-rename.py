#!/usr/bin/env python3
"""rename column"""

import pandas as pd


def rename(df):
    '''
    Renames Timestamp to Datetime, converts to datetime,
    and keeps only Datetime and Close columns.

    :param df: pd.DataFrame
    :return: modified pd.DataFrame
    '''
    # Rename Timestamp to Datetime
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the timestamp values to datatime values
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Display only the Datetime and Close column
    df = df[['Datetime', 'Close']]

    return df
