"""Database-backed, read-only dashboard for the all-US hourly BUY +3% research."""
from datetime import datetime, timedelta
from sqlalchemy import text
from core import database

TABLE='us_all_hourly_buy_trades_factors'

def ensure_dashboard_table():
    with database().begin() as s:
        s.execute(text(f'''CREATE TABLE IF NOT EXISTS {TABLE} (
            symbol TEXT NOT NULL, squaring_factor INTEGER NOT NULL, signal_utc TIMESTAMP NOT NULL,
            repeat_peak_high DOUBLE PRECISION, repeat_peak_bars INTEGER,
            entry_price DOUBLE PRECISION, gap_1h INTEGER, touches INTEGER,
            status TEXT NOT NULL, exit_utc TIMESTAMP, bars_to_exit INTEGER,
            bars_observed INTEGER, last_price DOUBLE PRECISION,
            realized_return_pct DOUBLE PRECISION, floating_return_pct DOUBLE PRECISION,
            combined_return_pct DOUBLE PRECISION, max_adverse_pct DOUBLE PRECISION,
            data_end_utc TIMESTAMP, PRIMARY KEY(symbol, squaring_factor, signal_utc))'''))

def publish_trades(csv_path):
    import pandas as pd
    ensure_dashboard_table()
    df=pd.read_csv(csv_path)
    df=df.astype(object).where(pd.notna(df),None)
    cols=('symbol','squaring_factor','repeat_peak_high','repeat_peak_bars','signal_utc','entry_price','gap_1h','touches','status','exit_utc',
          'bars_to_exit','bars_observed','last_price','realized_return_pct',
          'floating_return_pct','combined_return_pct','max_adverse_pct','data_end_utc')
    def dt(v):
        return datetime.strptime(str(v),'%Y-%m-%d %H:%M') if v not in (None,'') else None
    rows=[]
    for rec in df.to_dict('records'):
        rows.append({k:(dt(rec.get(k)) if k.endswith('_utc') else
                        int(rec[k]) if k in ('squaring_factor','repeat_peak_bars','gap_1h','touches','bars_to_exit','bars_observed') and rec.get(k) is not None else
                        rec.get(k)) for k in cols})
    with database().begin() as s:
        s.execute(text(f'DELETE FROM {TABLE}'))
        q=text(f'''INSERT INTO {TABLE} ({','.join(cols)}) VALUES ({','.join(':'+k for k in cols)})''')
        for i in range(0,len(rows),500): s.execute(q,rows[i:i+500])
    return len(rows)

def dashboard(days=60,status='ALL',query='',page=1,page_size=50,gap_min=None,gap_max=None,max_factor=7):
    ensure_dashboard_table()
    max_factor=int(max_factor)
    if not 2<=max_factor<=50: raise ValueError('Squaring factor must be between 2 and 50')
    status=status if status in ('ALL','EXIT','OPEN') else 'ALL'
    page=max(1,int(page));page_size=max(1,min(100,int(page_size)))
    if gap_min is not None: gap_min=max(0,min(10000,int(gap_min)))
    if gap_max is not None: gap_max=max(0,min(10000,int(gap_max)))
    if gap_min is not None and gap_max is not None and gap_min>gap_max:
        raise ValueError('الحد الأدنى للـ Gap أكبر من الحد الأقصى')
    with database()() as s:
        latest=s.execute(text(f'SELECT MAX(data_end_utc) FROM {TABLE}')).scalar()
        earliest=s.execute(text(f'SELECT MIN(signal_utc) FROM {TABLE}')).scalar()
        stored_max=s.execute(text(f'SELECT MAX(squaring_factor) FROM {TABLE}')).scalar()
        if latest is None:
            return {'rows':[],'total':0,'closed':0,'opened':0,'win_rate':None,'mean_return':None,
                    'realized':0,'floating':0,'avg_bars':None,'median_bars':None,
                    'latest':None,'earliest':None,'days':days,'page':page,'pages':0,'status':status,'query':query,'gap_min':gap_min,'gap_max':gap_max,'max_factor':max_factor,'stored_max':stored_max}
        params={'cutoff':latest-timedelta(days=days) if days else datetime(1970,1,1),
                'status':status,'pattern':'%'+query.strip().upper()[:20]+'%',
                'limit':page_size,'offset':(page-1)*page_size, 'gap_min':gap_min,'gap_max':gap_max,'max_factor':max_factor,'stored_max':stored_max}
        wh="squaring_factor BETWEEN 2 AND :max_factor AND signal_utc >= :cutoff AND (:status = 'ALL' OR status = :status) AND UPPER(symbol) LIKE :pattern AND (CAST(:gap_min AS INTEGER) IS NULL OR gap_1h >= CAST(:gap_min AS INTEGER)) AND (CAST(:gap_max AS INTEGER) IS NULL OR gap_1h <= CAST(:gap_max AS INTEGER))"
        stats=s.execute(text(f'''SELECT COUNT(*) total,
            COUNT(*) FILTER (WHERE status='EXIT') closed,
            COUNT(*) FILTER (WHERE status='OPEN') opened,
            COALESCE(SUM(realized_return_pct),0) realized,
            COALESCE(SUM(floating_return_pct),0) floating,
            AVG(combined_return_pct) mean_return,
            AVG(bars_to_exit) FILTER (WHERE status='EXIT') avg_bars,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bars_to_exit) FILTER (WHERE status='EXIT') median_bars
            FROM {TABLE} WHERE {wh}'''),params).mappings().one()
        result=s.execute(text(f'''SELECT symbol,squaring_factor,repeat_peak_high,repeat_peak_bars,signal_utc,entry_price,gap_1h,touches,status,exit_utc,
                    bars_to_exit,bars_observed,last_price,realized_return_pct,floating_return_pct,
                    combined_return_pct,max_adverse_pct,data_end_utc
                    FROM {TABLE} WHERE {wh} ORDER BY signal_utc DESC,symbol LIMIT :limit OFFSET :offset'''),params).mappings().all()
    total=int(stats['total']);closed=int(stats['closed'])
    return dict(rows=[dict(x) for x in result],total=total,closed=closed,opened=int(stats['opened']),
                win_rate=(100*closed/total if total else None),mean_return=stats['mean_return'],
                realized=stats['realized'],floating=stats['floating'],avg_bars=stats['avg_bars'],
                median_bars=stats['median_bars'],latest=latest,earliest=earliest,
                days=days,page=page,pages=(total+page_size-1)//page_size,status=status,query=query,gap_min=gap_min,gap_max=gap_max,max_factor=max_factor,stored_max=stored_max)
