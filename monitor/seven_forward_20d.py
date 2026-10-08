"""Forward 20-session excursions from Repeat price, beginning NEXT session.
Daily OHLC cannot resolve which level crossed first if both are hit in one bar.
"""
import math
import pandas as pd


def forward_20d(df, signal_date, repeat_price, horizon=20):
    empty = dict(max_rise_20d_pct=None, max_drawdown_20d_pct=None,
                 plus5_before_minus3="NO_DATA", plus5_before_minus5="NO_DATA",
                 first_hit_minus3=None, first_hit_minus5=None,
                 observed_20d_sessions=0, complete_20d=False)
    if df is None or df.empty or repeat_price is None or float(repeat_price) <= 0:
        return empty
    x = df.sort_values("date").reset_index(drop=True)
    ds = pd.to_datetime(x["date"]).dt.normalize()
    candidates = x.index[ds == pd.Timestamp(signal_date).normalize()].tolist()
    if not candidates:
        return empty
    entry = float(repeat_price)
    future = x.iloc[candidates[-1] + 1:candidates[-1] + 1 + horizon]
    if future.empty:
        return {**empty, "plus5_before_minus3":"PENDING", "plus5_before_minus5":"PENDING"}
    if "high" not in future or "low" not in future:
        return {**empty, "observed_20d_sessions":len(future), "complete_20d":len(future)>=horizon}
    highs = pd.to_numeric(future["high"], errors="coerce")
    lows = pd.to_numeric(future["low"], errors="coerce")
    valid = highs.notna() & lows.notna() & (highs >= lows) & (lows > 0)
    max_rise = (float(highs[valid].max()) / entry - 1) * 100 if valid.any() else None
    max_dd = (float(lows[valid].min()) / entry - 1) * 100 if valid.any() else None

    def first_cross(target, stop):
        for session, (hi,lo,ok) in enumerate(zip(highs, lows, valid), start=1):
            if not ok:
                return "NO_DATA", None
            hit_target = hi >= entry * 1.05
            hit_stop = lo <= entry * (1 - stop/100)
            if hit_target and hit_stop:
                return "AMBIGUOUS", session
            if hit_target:
                return "TARGET_FIRST", session
            if hit_stop:
                return "STOP_FIRST", session
        return ("PENDING" if len(future)<horizon else "NEITHER"), None

    r3, n3 = first_cross(5, 3)
    r5, n5 = first_cross(5, 5)
    return {"max_rise_20d_pct":round(max_rise,2) if max_rise is not None else None,
            "max_drawdown_20d_pct":round(max_dd,2) if max_dd is not None else None,
            "plus5_before_minus3":r3, "plus5_before_minus5":r5,
            "first_hit_minus3":n3, "first_hit_minus5":n5,
            "observed_20d_sessions":len(future),"complete_20d":len(future)>=horizon}
