"""Frozen exploratory Repeat + delayed-breakout screen rule, EGX and S&P500.
Not a validated forecast or trading recommendation.
"""
import pandas as pd
from monitor.seven_accumulation_study import _prices, features

RULE_LABEL = "Gap 11–20 | Touches 2 | Higher lows in previous 60 sessions"

def classify_repeat(frame, repeat_date, gap, touches):
    """Use only bars through the Repeat date; no forward leakage.

    Distinguish missing history from a failed rule. The higher-lows definition is
    the same as the research: minimum low in second 30 bars > first 30 bars.
    """
    x=_prices(frame)
    dates=pd.to_datetime(x.date).dt.normalize()
    idx=dates[dates == pd.Timestamp(repeat_date).normalize()].index
    result={'breakout_rule_match':False,'breakout_rule_status':'NO_DATA',
            'breakout_pre60_higher_lows':None,'breakout_pre60_volume_trend':None,
            'breakout_pre60_range_contraction':None}
    if not len(idx):return result
    f=features(x,int(idx[-1]))
    higher=f.get('pre60_higher_lows')
    result.update(breakout_pre60_higher_lows=higher,
                  breakout_pre60_volume_trend=f.get('pre60_volume_trend'),
                  breakout_pre60_range_contraction=f.get('pre60_range_contraction'))
    if higher is None:return result
    matched=(11 <= int(gap) <= 20 and int(touches)==2 and higher is True)
    result.update(breakout_rule_match=matched,breakout_rule_status='MATCH' if matched else 'NO_MATCH')
    return result
