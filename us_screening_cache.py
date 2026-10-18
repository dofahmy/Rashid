"""Persistent full-range screening snapshots, instant parametric view filters."""
import hashlib, json, os, time
from pathlib import Path

CACHE_ROOT=Path(os.environ.get('US_SCREENING_CACHE_DIR', '/data/us_screening_cache'))
CACHE_VERSION='us-only-v2'

def key_for(params):
    core={k:params[k] for k in ('factor_min','system_number','price_tolerance_pct','max_gap_sessions')}
    core['cache_version']=CACHE_VERSION
    return hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()[:32]

def snapshot_path(params): return CACHE_ROOT / ('snapshot_'+key_for(params)+'.json')

def available_snapshot(params):
    # A broader cached factor range can serve any narrower interval with matching
    # tolerance/gap. Never silently reuse incompatible signal calculations.
    best=None
    for path in CACHE_ROOT.glob('snapshot_*.json'):
        try:
            obj=json.loads(path.read_text())
            version=obj.get('cache_version')
            if version != CACHE_VERSION:
                # Recover old, unversioned snapshots ONLY if every stored ticker is
                # a valid US ticker, and the required future-closes data is present.
                # Do not use older mixed-market snapshots or incompatible versions.
                if version is not None: continue
                from us_seven_engine import is_us_symbol
                rows=obj.get('result',{}).get('rows',[])
                if not all(is_us_symbol(row.get('symbol')) and isinstance(row.get('_forward_closes'),list) for row in rows):
                    continue
                if 'params' not in obj or 'result' not in obj: continue
                obj['cache_version']=CACHE_VERSION
                # Persist the safe recovered snapshot so future loads are cheap.
                from us_screening_job import atomic_json
                atomic_json(path,obj)
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
    from us_seven_engine import is_us_symbol
    for original in snapshot['result'].get('rows',[]):
        if not is_us_symbol(original.get('symbol')): continue
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
        row={k:v for k,v in original.items() if k not in ('_forward_closes','_forward_highs')}
        closes=original.get('_forward_closes',[])
        n=int(params['return_sessions'])
        row['sessions_observed_after_repeat']=len(closes)
        row['horizon_close']=None
        row['horizon_return_pct']=None
        if len(closes)>=n and closes[n-1] is not None and float(row['repeat_price'])>0:
            row['horizon_close']=closes[n-1]
            row['horizon_return_pct']=round((closes[n-1]/float(row['repeat_price'])-1)*100,2)
        # First subsequent SESSION whose daily HIGH reaches the target.
        # No Repeat recomputation. Existing snapshots require one-time High backfill.
        target_pct=float(params.get('target_return_pct',3.0))
        repeat_price=float(row['repeat_price'])
        row['target_return_pct']=target_pct
        row['target_hit_sessions']=None
        if repeat_price > 0:
            target_high=repeat_price*(1+target_pct/100.0)
            highs=original.get('_forward_highs',[])
            for pos, high_value in enumerate(highs,1):
                if high_value is not None and float(high_value)>=target_high - 1e-10:
                    row['target_hit_sessions']=pos
                    break
        rows.append(row)
    result={k:v for k,v in snapshot['result'].items() if k!='rows'}
    result.update(rows=rows,count=len(rows),return_sessions=int(params['return_sessions']),
                  factor_min=int(params['factor_min']),system_number=int(params['system_number']),
                  factor_parity=params['factor_parity'],recent_days=params.get('recent_days'))
    return result
