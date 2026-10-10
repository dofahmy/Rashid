"""Independent DODZ backtest: factors 2/3, gap 1..3, signal UTC 19:30.
Read-only. No modification to original DODZ stored recommendations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from core import database
from monitor.seven_system import _single
from monitor.sp500_seven_hourly import _hourly_clusters, _rule
from monitor.sp500_seven_system import _repeat_direction_from_candle
from monitor.us_all_hourly_buy_study import TABLE as BARS

def signals_for_symbol(sym,df):
    if len(df)<61:return []
    df=df.sort_values('date').reset_index(drop=True)
    raw=[]
    for factor in (2,3):
        scored,_=_single(df,True,system_number=factor)
        for cluster in _hourly_clusters(scored,1.0,3):
            idx=cluster[-1][0]
            gap,touches,higher,match=_rule(df,cluster)
            # The shared MATCH includes a hard-coded 11..20 gap.
            # DODZ overrides only that range; preserves touches=2 and rising 60H lows.
            if not (1<=int(gap)<=3 and touches==2 and higher is True):continue
            stamp=pd.Timestamp(df.iloc[idx].date)
            if stamp.strftime('%H:%M')!='19:30':continue
            candle=df.iloc[idx].to_dict();candle['raw_close']=candle['close']
            direction,_=_repeat_direction_from_candle(candle)
            if direction!='BUY':continue
            entry=float(np.mean([x[2] for x in cluster]))
            if not np.isfinite(entry) or entry<=0:continue
            raw.append(dict(idx=idx,factor=factor,entry=entry,gap=int(gap),touches=int(touches)))
    # Different factors can produce same symbol/time repeat; execute one trade only.
    by_time={}
    for s in raw:by_time.setdefault(s['idx'],[]).append(s)
    events=[]; blocked_until=-1
    for idx, candidates in sorted(by_time.items()):
        if idx<=blocked_until: continue
        primary=min(candidates,key=lambda e:e['factor'])
        factor=primary['factor']; entry=primary['entry']; limit=factor*3
        post=df.iloc[idx+1:]
        deadline=idx+limit
        exit_idx=None;exit_price=None;reason=None
        for j in range(idx+1,min(len(df),deadline+1)):
            row=df.iloc[j]
            if pd.notna(row.high) and float(row.high)>=entry*1.03:
                exit_idx=j;exit_price=entry*1.03;reason='TARGET_3';break
            if j==deadline and pd.notna(row.close):
                exit_idx=j;exit_price=float(row.close);reason='TIME';break
        if exit_idx is not None:blocked_until=exit_idx
        else:blocked_until=len(df)
        end=exit_idx if exit_idx is not None else len(df)-1
        observed=df.iloc[idx+1:end+1]
        minlow=pd.to_numeric(observed.low,errors='coerce').min() if len(observed) else np.nan
        last=float(df.iloc[-1].close)
        actual_ret=(exit_price/entry-1)*100 if exit_idx is not None else None
        floating=(last/entry-1)*100 if exit_idx is None else None
        events.append(dict(symbol=sym,signal_utc=pd.Timestamp(df.iloc[idx].date).to_pydatetime(),
           squaring_factor=factor,factors=','.join(str(c['factor']) for c in sorted(candidates,key=lambda c:c['factor'])),
           entry_price=round(entry,6),gap_1h=primary['gap'],touches=primary['touches'],deadline_bars=limit,
           status='EXIT' if exit_idx is not None else 'OPEN',exit_reason=reason,
           exit_utc=pd.Timestamp(df.iloc[exit_idx].date).to_pydatetime() if exit_idx is not None else None,
           exit_price=round(exit_price,6) if exit_price is not None else None,
           bars_to_exit=exit_idx-idx if exit_idx is not None else None,
           observed_bars=len(df)-idx-1,last_price=last,
           realized_return_pct=round(actual_ret,5) if actual_ret is not None else None,
           floating_return_pct=round(floating,5) if floating is not None else None,
           combined_return_pct=round(actual_ret if actual_ret is not None else floating,5),
           max_adverse_pct=round((float(minlow)/entry-1)*100,5) if np.isfinite(minlow) else None,
           data_end_utc=pd.Timestamp(df.iloc[-1].date).to_pydatetime()))
    return events

def run(output_dir):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    trades=[]; last_sym=None; bars=[];count=0;errors=0
    def flush(sym,buf):
        nonlocal count,errors
        if not buf:return
        frame=pd.DataFrame(buf,columns=['date','open','high','low','close','volume'])
        frame['date']=pd.to_datetime(frame.date,utc=True).dt.tz_localize(None)
        for col in ('open','high','low','close','volume'):
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
        try:
            trades.extend(signals_for_symbol(sym,frame))
        except Exception as exc:
            errors+=1
            print(f'DODZ 2/3 error {sym}: {exc}',flush=True)
        count+=1
        if count%250==0: print(f'DODZ 2/3 scanned {count} symbols; trades={len(trades)}',flush=True)
    with database()() as s:
        for r in s.execute(text(f'SELECT symbol,bar_time,open,high,low,close,volume FROM {BARS} ORDER BY symbol,bar_time')):
            if last_sym is not None and r[0]!=last_sym:
                flush(last_sym,bars);bars=[]
            last_sym=r[0];bars.append(r[1:])
        if last_sym is not None:flush(last_sym,bars)
    columns=('symbol','signal_utc','squaring_factor','factors','entry_price','gap_1h','touches','deadline_bars','status','exit_reason','exit_utc','exit_price','bars_to_exit','observed_bars','last_price','realized_return_pct','floating_return_pct','combined_return_pct','max_adverse_pct','data_end_utc')
    df=pd.DataFrame(trades,columns=columns)
    trades_file=out/'dodz_2_3_gap_1_3_trades.csv'
    df.to_csv(trades_file,index=False,encoding='utf-8-sig')
    def stats(sub):
        n=len(sub)
        target=int((sub.exit_reason=='TARGET_3').sum()) if n else 0
        timed=int((sub.exit_reason=='TIME').sum()) if n else 0
        opened=int((sub.status=='OPEN').sum()) if n else 0
        # Separate open/closed reporting, preserve real end-of-study marks for OPEN.
        exit_df=sub[sub.status=='EXIT'] if n else sub
        return dict(total=n,target=target,time=timed,open=opened,
            target_rate_pct=round(target/n*100,2) if n else None,
            avg_return_all_pct=round(float(sub.combined_return_pct.mean()),4) if n else None,
            avg_return_closed_pct=round(float(exit_df.realized_return_pct.mean()),4) if len(exit_df) else None,
            median_return_all_pct=round(float(sub.combined_return_pct.median()),4) if n else None,
            worst_return_pct=round(float(sub.combined_return_pct.min()),4) if n else None,
            best_return_pct=round(float(sub.combined_return_pct.max()),4) if n else None)
    breakdown=[]
    for factor in (2,3):
        for gap in (1,2,3):
            block=df[(df.squaring_factor==factor)&(df.gap_1h==gap)]
            breakdown.append(dict(factor=factor,gap=gap,**stats(block)))
    by_group=out/'dodz_2_3_gap_1_3_breakdown.csv'
    pd.DataFrame(breakdown).to_csv(by_group,index=False,encoding='utf-8-sig')
    summary=dict(symbols_scanned=count,errors=errors,criteria={'squaring_factors':[2,3],'gap_min':1,'gap_max':3,'signal_utc':'19:30','target_pct':3.0,'deadline_bars':'factor * 3'},all=stats(df),factor_2=stats(df[df.squaring_factor==2]),factor_3=stats(df[df.squaring_factor==3]),files=[str(trades_file),str(by_group)],note='Backtest using cached 1H bars and hypothetical fills at Repeat Price; no fees/spread/slippage, opens marked to last saved bar. Unlike filtering old DODZ results, recomputes one-position-per-symbol with only these factors and gaps.')
    (out/'dodz_2_3_gap_1_3_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Factor  Gap  Trades  Target  Time  Open  Target%   Average%')
    for x in breakdown:
        print(f"{x['factor']:>6} {x['gap']:>4} {x['total']:>7} {x['target']:>7} {x['time']:>5} {x['open']:>5} {str(x['target_rate_pct']):>8} {str(x['avg_return_all_pct']):>10}")
    return summary

def main():
    p=argparse.ArgumentParser(description='Isolated DODZ 2/3 factors, Gap 1..3 research')
    p.add_argument('--output-dir',default='/tmp/dodz-2-3-gap-1-3')
    args=p.parse_args()
    run(args.output_dir)

if __name__=='__main__': main()
