"""Read-only US Time-Price historical backtest from Rajih PostgreSQL.

Run from project root: python us_tp_backtest.py --out /tmp/us_tp_test
No writes to PostgreSQL, no trading publications, no Telegram calls.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from sqlalchemy import text
from core import database

TYPES = ('DIFFERENT_ANGLES', 'SAME_ANGLE_DIFF_ANCHORS', 'SAME_LINE_TWICE')
GROUP_SQL = text('''
 SELECT c.day,c.symbol,c.signal_type,c.anchor_type,c.status,c.angles,c.close_price,
        g.stock_count
 FROM us_tp_daily_candidates c
 JOIN (
   SELECT day,signal_type,anchor_type,status,COUNT(DISTINCT symbol) AS stock_count
   FROM us_tp_daily_candidates
   GROUP BY day,signal_type,anchor_type,status
   HAVING COUNT(DISTINCT symbol)>=:minimum
 ) g ON g.day=c.day AND g.signal_type=c.signal_type
     AND g.anchor_type=c.anchor_type AND g.status=c.status
 ORDER BY c.symbol,c.day,c.signal_type,c.anchor_type,c.status
''')


def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def summarise(rows):
    closed = [r for r in rows if r['outcome'] in ('TP', 'TIME_EXIT')]
    tp = [r for r in rows if r['outcome'] == 'TP']
    forced = [r for r in rows if r['outcome'] == 'TIME_EXIT']
    returns = [r['return_pct'] for r in closed]
    return {
        'signals':len(rows), 'closed':len(closed),'open_or_incomplete':len(rows)-len(closed),
        'tp_hits':len(tp), 'time_exits':len(forced),
        'tp_rate_pct_closed':round(100*len(tp)/len(closed),2) if closed else None,
        'avg_return_pct':round(mean(returns),4) if returns else None,
        'median_return_pct':round(median(returns),4) if returns else None,
        'worst_return_pct':round(min(returns),4) if returns else None,
        'win_rate_pct':round(100*sum(x>0 for x in returns)/len(returns),2) if returns else None,
        'avg_holding_sessions':round(mean(r['exit_session'] for r in closed),2) if closed else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/tmp/us_tp_backtest', help='Local file prefix (no DB writes)')
    p.add_argument('--min-stocks',type=int,default=5)
    p.add_argument('--max-sessions',type=int,default=20)
    p.add_argument('--target-pct',type=float,default=3.0)
    a=p.parse_args()
    if a.min_stocks<2 or a.max_sessions<1 or a.target_pct<=0:
        p.error('invalid settings')
    DB=database()
    with DB() as s:
        candidates=[dict(x) for x in s.execute(GROUP_SQL, {'minimum':a.min_stocks}).mappings()]
    if not candidates:
        print('No groups qualified. No changes made.');return
    by_symbol=defaultdict(list)
    for c in candidates: by_symbol[c['symbol']].append(c)

    outcomes=[]
    missing_symbols=[]
    # One SQL query per symbol, instead of tens of thousands of per-signal queries.
    for i,(symbol,signals) in enumerate(by_symbol.items(),1):
        first_day=min(sg['day'] for sg in signals)
        with DB() as s:
            bars=[dict(x) for x in s.execute(text('''
                SELECT day,h,c FROM us_tp_daily_bars
                WHERE symbol=:symbol AND day>:first_day ORDER BY day
            '''),{'symbol':symbol,'first_day':first_day}).mappings()]
        if not bars: missing_symbols.append(symbol)
        for sg in signals:
            future=[b for b in bars if b['day']>sg['day']][:a.max_sessions]
            entry=float(sg['close_price'])
            target=entry*(1+a.target_pct/100)
            result=dict(sg)
            result.update(entry=round(entry,6),target=round(target,6),
                          available_following_sessions=len(future),outcome='OPEN_INCOMPLETE',
                          exit_day='',exit_session='',exit_price='',return_pct='')
            if not future:
                result['outcome']='NO_FUTURE_BARS'
            else:
                for pos, b in enumerate(future,1):
                    if float(b['h'])>=target:
                        result.update(outcome='TP',exit_day=b['day'],exit_session=pos,
                                      exit_price=round(target,6),return_pct=round(a.target_pct,6))
                        break
                else:
                    if len(future)>=a.max_sessions:
                        end=future[-1]
                        ret=(float(end['c'])/entry-1)*100
                        result.update(outcome='TIME_EXIT',exit_day=end['day'],exit_session=a.max_sessions,
                                      exit_price=round(float(end['c']),6),return_pct=round(ret,6))
            outcomes.append(result)
        if i%100==0:print('Processed symbols:',i,'/',len(by_symbol),flush=True)

    groups={}
    for r in candidates:
        k=(r['day'],r['signal_type'],r['anchor_type'],r['status'])
        groups[k]=max(groups.get(k,0),int(r['stock_count']))
    by_type=defaultdict(list)
    by_status=defaultdict(list)
    for r in outcomes:
        by_type[r['signal_type']].append(r)
        by_status[(r['signal_type'],r['anchor_type'],r['status'])].append(r)
    type_results=[{'signal_type':t, **summarise(by_type[t])} for t in TYPES]
    type_status_results=[{'signal_type':t,'anchor_type':anc,'status':status,**summarise(v)}
                         for (t,anc,status),v in sorted(by_status.items())]
    report={
        'parameters':{'min_stocks':a.min_stocks,'target_pct':a.target_pct,
                      'max_sessions':a.max_sessions,'entry':'signal_day_close',
                      'tp':'first later-session HIGH >= target; fill at target',
                      'time_exit':'session 20 CLOSE if TP never reached',
                      'commission_slippage':'not included'},
        'data_scope':{'candidate_rows':len(candidates),'qualified_groups':len(groups),
                      'distinct_symbols':len(by_symbol),'missing_future_bar_symbols':len(missing_symbols)},
        'all':summarise(outcomes), 'by_type':type_results,
        'warnings':[
          'Partial US universe: qualification can change when new symbols are scanned.',
          'Historical selection bias: groups use the current saved universe, not necessarily symbols available at each historical date.',
          'Symbols with missing sessions may cause a misleading twentieth-session count; check OHLC coverage.',
          'TP within twenty sessions counts as closed even if later data unavailable; unhit positions with fewer than twenty sessions remain open.',
          'No transaction costs, slippage, spread, delisting data, or adjusted execution model.',
          'Signal-day bar HIGH is deliberately excluded from TP tests.'
        ]}
    prefix=Path(a.out)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    save_csv(Path(str(prefix)+'_trades.csv'),outcomes,
             ['day','symbol','signal_type','anchor_type','status','angles','stock_count','close_price','entry','target',
              'available_following_sessions','outcome','exit_day','exit_session','exit_price','return_pct'])
    save_csv(Path(str(prefix)+'_summary.csv'),type_results,
             ['signal_type','signals','closed','open_or_incomplete','tp_hits','time_exits','tp_rate_pct_closed',
              'avg_return_pct','median_return_pct','worst_return_pct','win_rate_pct','avg_holding_sessions'])
    save_csv(Path(str(prefix)+'_by_status.csv'),type_status_results,
             ['signal_type','anchor_type','status','signals','closed','open_or_incomplete','tp_hits','time_exits',
              'tp_rate_pct_closed','avg_return_pct','median_return_pct','worst_return_pct','win_rate_pct','avg_holding_sessions'])
    Path(str(prefix)+'_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'groups':len(groups),'all':report['all'],'by_type':type_results},indent=2,ensure_ascii=False))
    print('Files:',str(prefix)+'_trades.csv',str(prefix)+'_summary.csv',str(prefix)+'_by_status.csv',str(prefix)+'_report.json')
    print('READ-ONLY: No DB modifications or alerts.')

if __name__=='__main__':main()
