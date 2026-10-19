#!/usr/bin/env python3
"""Summarize 90-session return distribution from existing seven RP Mirror outputs. Stdlib only."""
import argparse,csv,json,math,statistics
from pathlib import Path

BANDS=[('below_0',lambda x:x<0),('0_to_10',lambda x:0<=x<10),('10_to_20',lambda x:10<=x<20),('20_to_30',lambda x:20<=x<30),('30_to_50',lambda x:30<=x<50),('50_to_75',lambda x:50<=x<75),('75_to_100',lambda x:75<=x<100),('100_plus',lambda x:x>=100)]
REF_ORDER=['YDKG','BGMS','ZIM','FROG','MU','IMSR','IMNM']

def num(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError):return None

def med(a): return round(statistics.median(a),4) if a else None
def mean(a):return round(statistics.mean(a),4) if a else None

def summary(ref,window,rows):
    valid=[r for r in rows if num(r.get('max_high_return_90_pct')) is not None]
    highs=[num(r['max_high_return_90_pct']) for r in valid]
    closes=[num(r.get('close_90_pct')) for r in valid if num(r.get('close_90_pct')) is not None]
    lows=[num(r.get('worst_low_90_pct')) for r in valid if num(r.get('worst_low_90_pct')) is not None]
    winners=[r for r in valid if num(r['max_high_return_90_pct'])>=100]
    nonwinners=[r for r in valid if num(r['max_high_return_90_pct'])<100]
    first=[num(r.get('first_hit100_session')) for r in winners if num(r.get('first_hit100_session')) is not None]
    first30=[r for r in valid if num(r.get('max_high_return_first30_pct')) is not None and num(r['max_high_return_first30_pct'])>=100]
    unique_stocks=len({r['symbol'] for r in valid})
    out={'reference':ref,'rp_window':window,'cases':len(valid),'distinct_stocks':unique_stocks,
         'reached_100':len(winners),'not_reached_100':len(nonwinners),
         'hit100_rate_pct':round(100*len(winners)/len(valid),4) if valid else None,
         'hit100_first30_same_cohort':len(first30),'median_first_hit100_session':med(first),
         'mean_max_high_90_pct':mean(highs),'median_max_high_90_pct':med(highs),
         'mean_max_high_nonwinners_pct':mean([num(r['max_high_return_90_pct']) for r in nonwinners]),
         'median_max_high_nonwinners_pct':med([num(r['max_high_return_90_pct']) for r in nonwinners]),
         'mean_close_90_pct':mean(closes),'median_close_90_pct':med(closes),
         'mean_worst_low_90_pct':mean(lows),'median_worst_low_90_pct':med(lows),
         'close_90_negative':sum(x<0 for x in closes),'close_90_available':len(closes),
         'high_below_0':sum(x<0 for x in highs)}
    for name,func in BANDS:out[name]=sum(func(x) for x in highs)
    return out

def write(path,rows,cols):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='/data/seven_rp_mirrors_90')
    ap.add_argument('--out',default='/data/seven_rp_mirrors_90/distribution')
    a=ap.parse_args();root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    all_s=[]; missing=[]; details=[]
    for ref in REF_ORDER:
        folder=root/ref
        for file in sorted(folder.glob(f'{ref}_RP*_mirrors.csv')):
            window=int(file.stem.split('_RP')[1].split('_')[0]);
            with file.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            needed={'symbol','signal_date','max_high_return_90_pct','close_90_pct','worst_low_90_pct'}
            if rows and needed-set(rows[0]):raise RuntimeError(f'Missing columns in {file}: {needed-set(rows[0])}')
            s=summary(ref,window,rows);all_s.append(s)
            for r in rows:
                h=num(r.get('max_high_return_90_pct'))
                if h is None: continue
                band=next(name for name,fn in BANDS if fn(h))
                details.append({'reference':ref,'rp_window':window,'symbol':r['symbol'], 'signal_date':r.get('signal_date'), 'max_high_return_90_pct':h,'band':band,'close_90_pct':r.get('close_90_pct'),'worst_low_90_pct':r.get('worst_low_90_pct'),'first_hit100_session':r.get('first_hit100_session')})
        if not folder.exists():missing.append(ref)
    if not all_s:raise SystemExit(f'No *_mirrors.csv found under {root}.')
    cols=list(all_s[0]);write(out/'all_40_RP_windows_distribution.csv',all_s,cols)
    # Max hit100-rate among rows with at least one observation, tie-break more hits then more cases
    best=[]
    for ref in REF_ORDER:
        cand=[s for s in all_s if s['reference']==ref and s['cases']]
        if cand:best.append(max(cand,key=lambda s:(s['hit100_rate_pct'],s['reached_100'],s['cases'])))
    write(out/'best_RP_per_reference_distribution.csv',best,cols)
    dcols=list(details[0]) if details else []
    if dcols:write(out/'all_matched_cases_with_bands.csv',details,dcols)
    report={'status':'complete','input':str(root),'output':str(out),'windows':len(all_s),'best_references':len(best),'matched_case_rows':len(details),'missing_reference_folders':missing,
      'notes':['Counts are stock-date signals, not independent stocks.','Bands are mutually exclusive and use maximum future daily High within the next 90 sessions, NOT day-90 close.','Signal day is excluded; source CSV assumes full 90 future stock sessions.','Negative max-High is possible if all 90 subsequent High values are below entry close.','Close-90 and worst-low give context on drawdown and exit outcome, not a realizable +100% execution price.','Best RP chosen using historical data; result is in-sample, not a prospective trading accuracy estimate.']}
    (out/'distribution_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',json.dumps(report,ensure_ascii=False))
    print('\nBEST RP PER STRATEGY — each bin is exclusive:')
    headline=['reference','rp_window','cases','not_reached_100','below_0','0_to_10','10_to_20','20_to_30','30_to_50','50_to_75','75_to_100','100_plus','median_max_high_nonwinners_pct','median_close_90_pct']
    print(' | '.join(headline))
    for s in best:print(' | '.join(str(s.get(k,'')) for k in headline))
    print('\nFILES:',out/'best_RP_per_reference_distribution.csv',out/'all_40_RP_windows_distribution.csv',out/'all_matched_cases_with_bands.csv')

if __name__=='__main__':main()
