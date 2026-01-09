#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Preprocessing
df.drop(columns=['Weighted_Price'], inplace=True)
df.rename(columns={'Timestamp': "Date"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
cols = ['Volume_(BTC)', 'Volume_(Currency)']
df[cols] = df[cols].fillna(0)
df = df[df.index.year >= 2017]
hi = df['High'].groupby(df.index.date).max()
lo = df['Low'].groupby(df.index.date).min()
opn = df['Open'].groupby(df.index.date).mean()
close = df['Close'].groupby(df.index.date).mean()
vol_btc = df['Volume_(BTC)'].groupby(df.index.date).sum()
vol_cur = df['Volume_(Currency)'].groupby(df.index.date).sum()
df = pd.DataFrame({
    'High': hi,
    'Low': lo,
    'Open': opn,
    'Close': close,
    'Volume_(BTC)': vol_btc,
    'Volume_(Currency)': vol_cur
})

# Print before Plot
print(df)

# Plotting
df.plot(figsize=(14, 6))
plt.xlabel('Date')
plt.show()