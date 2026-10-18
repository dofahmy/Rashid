"""Persistent full-range screening snapshots, instant parametric view filters."""
import hashlib, json, os, time
from pathlib import Path

CACHE_ROOT=Path(os.environ.get('US_SCREENING_CACHE_DIR', '/data/us_screening_cache'))

def key_for(params):
    core={k:params[k] for k in ('factor_min','system_number','price_tolerance_pct','max_gap_sessions')}
    return hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()[:32]

def snapshot_path(params): return CACHE_ROOT / ('snapshot_'+key_for(params)+'.json')

def available_snapshot(params):
    # A broader cached factor range can serve any narrower interval with matching
    # tolerance/gap. Never silently reuse incompatible signal calculations.
    best=None
    for path in CACHE_ROOT.glob('snapshot_*.json'):
        try:
            obj=json.loads(path.read_text())
            p=obj['params']
            if (float(p['price_tolerance_pct'])==float(params['price_tolerance_pct'])
                and int(p['max_gap_sessions'])==int(params['max_gap_sessions'])
                and int(p['factor_min']) <= int(params['factor_min'])
                and int(p['system_number']) >= int(params['system_number'])):
                if best is None or (int(p['system_number'])-int(p['factor_min']) < int(best[1]['params']['system_number'])-int(best[1]['params']['factor_min'])):
                    best=(path,obj)
        except (OSError,ValueError,KeyError,TypeError):pass
    return best[1] if best else None

def filtered(snapshot, params, rule_only):
    rows=[]
    market_date=snapshot['result'].get('latest_market_date')
    from datetime import date
    for original in snapshot['result'].get('rows',[]):
        f=int(original['squaring_factor'])
        if not int(params['factor_min']) <= f <= int(params['system_number']):continue
        parity=params.get('factor_parity','both')
        if parity=='odd' and f%2==0:continue
        if parity=='even' and f%2!=0:continue
        if rule_only and not original.get('breakout_rule_match'):continue
        recent=params.get('recent_days')
        if recent is not None and market_date:
            age=(date.fromisoformat(market_date)-date.fromisoformat(original['latest_repeat_date'])).days
            if age>int(recent):continue
        row={k:v for k,v in original.items() if k!='_forward_closes'}
        closes=original.get('_forward_closes',[])
        n=int(params['return_sessions'])
        row['sessions_observed_after_repeat']=len(closes)
        row['horizon_close']=None
        row['horizon_return_pct']=None
        if len(closes)>=n and closes[n-1] is not None and float(row['repeat_price'])>0:
            row['horizon_close']=closes[n-1]
            row['horizon_return_pct']=round((closes[n-1]/float(row['repeat_price'])-1)*100,2)
        rows.append(row)
    result={k:v for k,v in snapshot['result'].items() if k!='rows'}
    result.update(rows=rows,count=len(rows),return_sessions=int(params['return_sessions']),
                  factor_min=int(params['factor_min']),system_number=int(params['system_number']),
                  factor_parity=params['factor_parity'],recent_days=params.get('recent_days'))
    return result
