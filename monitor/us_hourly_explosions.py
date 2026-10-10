"""Market-wide discovery of hourly >100% low-to-later-high moves within 250 candles.

Read-only PostgreSQL scan; independent from DODZ, Repeat, or other signals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from core import database

BARS = 'us_all_seven_hourly_bars'
FIELDS = ['symbol','start_utc','peak_utc','start_low','peak_high',
          'rise_pct','bars_to_peak','start_bar_index','peak_bar_index',
          'data_start_utc','data_end_utc','possible_data_issue']


def _dt(value):
    return value.isoformat(sep=' ') if isinstance(value,datetime) else str(value)


def analyze_symbol(symbol, bars, window, min_pct):
    """O(N) monotone queue of preceding lows; exclude same-bar lows.

    Breakout episodes consist of consecutive candles with >threshold movement.
    One best candidate (largest %) represents each episode. Overall best
    across all episodes also reported separately.
    """
    if len(bars)<2:
        return [], None
    lows=deque()   # (index, low), always ascending by low
    episodes=[]
    best=None
    active=None
    starts=[r[0] for r in bars]
    for i, (time, low, high) in enumerate(bars):
        # candidate lowest earlier low at most `window` bars behind
        while lows and lows[0][0] < i-window:
            lows.popleft()
        event=None
        if lows and high is not None and high>0 and math.isfinite(high):
            start_idx,start_low=lows[0]
            gain=100*(high/start_low-1)
            if gain>min_pct:  # strictly greater, as requested
                event=(symbol,starts[start_idx],time,start_low,high,gain,
                       i-start_idx,start_idx,i,starts[0],starts[-1])
        if event is None:
            if active is not None:
                episodes.append(active)
                active=None
        else:
            if active is None or event[5]>active[5]:
                active=event
            if best is None or event[5]>best[5]:
                best=event
        if low is not None and low>0 and math.isfinite(low):
            # Keep older position when low equal to favor larger elapsed duration.
            while lows and lows[-1][1]>low:
                lows.pop()
            lows.append((i,low))
    if active is not None:
        episodes.append(active)
    return episodes,best


def to_record(item):
    sym,st,en,low,high,gain,bars,si,ei,first,last=item
    return dict(symbol=sym,start_utc=_dt(st),peak_utc=_dt(en),
                start_low=round(low,8),peak_high=round(high,8),
                rise_pct=round(gain,4),bars_to_peak=bars,
                start_bar_index=si,peak_bar_index=ei,
                data_start_utc=_dt(first),data_end_utc=_dt(last),
                possible_data_issue='CHECK_SPLIT_OR_BAD_BARS' if gain>500 else '')


def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir,window=250,min_pct=100.0):
    if window<1 or window>10000:
        raise ValueError('window must be 1..10000')
    if min_pct<0:
        raise ValueError('min_pct cannot be negative')
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    count=0; invalid=0; best_by_symbol=[]; all_episodes=[]
    def process(sym,rows):
        nonlocal count,invalid
        if not rows:return
        eps,best=analyze_symbol(sym,rows,window,min_pct)
        count+=1
        if best is not None:best_by_symbol.append(to_record(best))
        all_episodes.extend(to_record(r) for r in eps)
        if count%250==0:
            print(f'Explosions scanned {count} symbols; qualifying={len(best_by_symbol)}; episodes={len(all_episodes)}',flush=True)

    with database()() as db:
        cursor=db.execute(text(f'SELECT symbol,bar_time,low,high FROM {BARS} ORDER BY symbol,bar_time'))
        last_symbol=None;current=[]
        for symbol,time,low,high in cursor:
            if last_symbol is not None and symbol!=last_symbol:
                process(last_symbol,current);current=[]
            last_symbol=symbol
            try:
                lo=float(low) if low is not None else None
                hi=float(high) if high is not None else None
                if lo is not None and (not math.isfinite(lo) or lo<=0):lo=None
                if hi is not None and (not math.isfinite(hi) or hi<=0):hi=None
                if lo is not None and hi is not None and hi<lo:
                    invalid+=1;lo=None;hi=None
                current.append((time,lo,hi))
            except (ValueError,TypeError):
                invalid+=1;current.append((time,None,None))
        if last_symbol is not None:process(last_symbol,current)

    best_by_symbol.sort(key=lambda x:x['rise_pct'],reverse=True)
    all_episodes.sort(key=lambda x:x['rise_pct'],reverse=True)
    p1=out/'us_explosions_best_per_symbol.csv'
    p2=out/'us_explosions_all_episodes.csv'
    save_csv(p1,best_by_symbol)
    save_csv(p2,all_episodes)
    summary={'symbols_scanned':count,'symbols_with_over_100pct':len(best_by_symbol),
             'episodes_over_100pct':len(all_episodes),'invalid_bars':invalid,
             'window_bars':window,'threshold_pct_strictly_above':min_pct,
             'definition':'earlier hourly Low -> later hourly High, up to N intervening hourly candles; both in stored market hours',
             'note':'Independent from DODZ. Peak highs may not be executable; splits, reverse splits, corporate actions and bar errors require checking.',
             'files':[str(p1),str(p2)]}
    (out/'us_explosions_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(f"{'Symbol':10} {'Rise %':>12} {'Bars':>6} {'Start (UTC)':20} {'Peak (UTC)':20}")
    for row in best_by_symbol[:50]:
        print(f"{row['symbol']:10} {row['rise_pct']:>11.2f}% {row['bars_to_peak']:>6} {row['start_utc']:20} {row['peak_utc']:20}")
    return summary


def main():
    p=argparse.ArgumentParser(description='All-US hourly >100% rally search, no DODZ filters')
    p.add_argument('--window-bars',type=int,default=250)
    p.add_argument('--min-rise-pct',type=float,default=100.0)
    p.add_argument('--output-dir',default='/tmp/us-explosions-100pct')
    a=p.parse_args()
    run(a.output_dir,a.window_bars,a.min_rise_pct)

if __name__=='__main__':main()
