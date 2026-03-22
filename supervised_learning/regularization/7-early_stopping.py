#!/usr/bin/env python3
"""7-early_stopping.py"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Earlystopping"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count != patience:
        return False, count
    else:
        return True, count
