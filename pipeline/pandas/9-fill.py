#!/usr/bin/env python3
""" Full Func"""


def fill(df):
    """
    input -> df
    output -> df preprocessed null values
    """

    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].ffill()
    df.loc[df['High'].isnull(), 'High'] = df['Close']
    df.loc[df['Open'].isnull(), 'Open'] = df['Close']
    df.loc[df['Low'].isnull(), 'Low'] = df['Close']
    df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
    return df