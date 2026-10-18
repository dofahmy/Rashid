"""One-time repair for existing US screening snapshots; no Repeat recomputation."""
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sqlalchemy import MetaData, Table, select
from core import database
from us_screening_cache import CACHE_ROOT
from us_screening_job import atomic_json


def main():
    paths = sorted(CACHE_ROOT.glob('snapshot_*.json'))
    if not paths:
        print('NO SNAPSHOTS FOUND')
        return
    db = database()
    with db() as s:
        t = Table('market_candles_1d', MetaData(), autoload_with=s.get_bind())
        for needed in ('symbol', 'session_date', 'h'):
            if needed not in t.c: raise RuntimeError('Missing DB column: '+needed)
        for path in paths:
            obj=json.loads(path.read_text(encoding='utf-8'))
            rows=obj.get('result',{}).get('rows',[])
            todo=[r for r in rows if not isinstance(r.get('_forward_highs'),list)]
            if not todo:
                print(path.name, 'already completed', len(rows))
                continue
            grouped=defaultdict(list)
            for row in todo:
                symbol=str(row.get('symbol','')).upper()
                date=str(row.get('latest_repeat_date',''))
                if not symbol or not date:continue
                grouped[symbol].append(row)
            updated=0
            for symbol,items in grouped.items():
                earliest=min(str(r['latest_repeat_date']) for r in items)
                candles=s.execute(select(t.c.session_date,t.c.h).where(t.c.symbol==symbol, t.c.session_date>earliest).order_by(t.c.session_date)).all()
                for row in items:
                    date=str(row['latest_repeat_date'])
                    closes=row.get('_forward_closes',[])
                    n=len(closes)
                    matches=[float(high) if high is not None else None for d,high in candles if str(d)[:10]>date]
                    # Match the length of forward closes (date-order as saved by engine).
                    row['_forward_highs']=matches[:n]
                    updated+=1
            atomic_json(path,obj)
            print('UPDATED',path.name,'rows:',updated,'total:',len(rows),flush=True)

if __name__=='__main__': main()
