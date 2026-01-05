#!/usr/bin/env python3
"""rename column"""

import pandas as pd


def rename(df):
    """RENAME THE COLUMN"""

    df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df.loc[:, ['Datetime', 'Close']]
    return df