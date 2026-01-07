#!/usr/bin/env python3
"""REVERSING AND FLIPPIN"""


def flip_switch(df):
  """
  input -> df
  output -> sorted, transposed df
  """

  df = df.sort_values('Timestamp', ascending=False).T
  return df
