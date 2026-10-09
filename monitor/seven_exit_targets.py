"""Historical first-touch profit exits for Repeat Price + Time, both markets.

This is a level-touch simulation, not a fill-price or realizable-P&L claim.
No stop-loss, fees, liquidity or slippage is assumed. Events start NEXT session.
"""
import pandas as pd


def repeat_exit_targets(df, signal_date, repeat_price, targets=(30, 50), exact_timestamp=False):
    out = {}
    for target in targets:
        out.update({f'exit_{target}_status': 'NO_DATA',
                    f'exit_{target}_session': None,
                    f'exit_{target}_date': None})
    out['exit_observed_sessions'] = 0
    out['exit_max_rise_pct'] = None
    out['repeat_peak_high'] = None
    out['repeat_peak_bars'] = None
    if df is None or df.empty or repeat_price is None:
        return out
    try:
        price = float(repeat_price)
    except (TypeError, ValueError):
        return out
    if price <= 0 or not {'date', 'high'}.issubset(df.columns):
        return out
    ordered = df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(ordered['date'], errors='coerce')
    target_date = pd.Timestamp(signal_date)
    matches = ordered.index[dates == target_date if exact_timestamp else dates.dt.normalize() == target_date.normalize()].tolist()
    if not matches:
        return out
    future = ordered.iloc[matches[-1] + 1:]
    if future.empty:
        for t in targets:
            out[f'exit_{t}_status'] = 'OPEN'
        return out
    highs = pd.to_numeric(future['high'], errors='coerce')
    out['exit_observed_sessions'] = len(future)
    valid = highs.notna() & (highs > 0)
    if valid.any():
        out['exit_max_rise_pct'] = round((float(highs[valid].max()) / price - 1) * 100, 2)
        peak_idx = highs[valid].idxmax()
        out['repeat_peak_high'] = round(float(highs.loc[peak_idx]), 4)
        out['repeat_peak_bars'] = int(future.index.get_loc(peak_idx)) + 1
    for target in targets:
        threshold = price * (1 + target / 100)
        for i, (rowidx, high) in enumerate(highs.items(), 1):
            if pd.isna(high) or high <= 0:
                out[f'exit_{target}_status'] = 'NO_DATA'
                break
            if high >= threshold:
                out[f'exit_{target}_status'] = 'EXIT'
                out[f'exit_{target}_session'] = i
                out[f'exit_{target}_date'] = dates.iloc[rowidx].isoformat(sep=' ', timespec='minutes') if exact_timestamp else dates.iloc[rowidx].date().isoformat()
                break
        else:
            out[f'exit_{target}_status'] = 'OPEN'
    return out
