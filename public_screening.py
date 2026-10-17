"""Isolated public EGX screening application. Deploy as a SEPARATE Railway service.
No admin routes, no password, no reliance on the existing Flask app.py.
Reuses monitor.seven_system.py and core.database() from the same repository.
"""
import os
import time
import subprocess
import sys
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from flask import Flask, abort, render_template, request, redirect, url_for, Response
from sqlalchemy import text
from core import database

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 8192
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Small in-memory per-process guard; run exactly one web worker for this service.
_buckets = defaultdict(deque)
_guard = threading.Lock()
_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 300
SEARCH_LIMIT_PER_MINUTE = 6
REFRESH_COOLDOWN_MINUTES = 30

@app.after_request
def security_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    return resp

def _rate_limit(key, limit, window=60):
    now = time.monotonic()
    with _guard:
        d = _buckets[key]
        while d and now - d[0] > window: d.popleft()
        if len(d) >= limit: return False
        d.append(now)
        if len(_buckets)>4000:
            stale = [k for k, v in _buckets.items() if not v or now-v[-1] > window]
            for k in stale: _buckets.pop(k, None)
        return True

def _public_ip():
    # Do not trust forged X-Forwarded-For for administrative decisions.
    # Refresh limiting is DB-global, regardless of IP.
    return request.remote_addr or 'unknown'

def _num(name, default, minimum, maximum, typ=int):
    try: value = typ(request.args.get(name, default))
    except (TypeError, ValueError): value = default
    return max(minimum, min(value, maximum))

@app.get('/health')
def health():
    return {'ok': True, 'service': 'public-screening'}

@app.get('/')
@app.get('/screening')
def screening():
    fmin = _num('factor_min', 7, 2, 1000)
    fmax = _num('system_number', 21, fmin, 1000)
    parity = request.args.get('factor_parity','both').strip().lower()
    if parity not in ('both', 'odd', 'even'): parity = 'both'
    tol = _num('price_tolerance_pct', 1.0, 0, 20, float)
    gap = _num('max_gap_sessions', 17, 1, 252)
    horizon = _num('return_sessions', 60, 1, 3000)
    recent_raw = request.args.get('recent_days','all').strip().lower()
    recent = None if recent_raw in ('all','none','') else _num('recent_days',90,1,3650)
    rule_only = request.args.get('rule_only','1') == '1'

    args = dict(factor_min=fmin, system_number=fmax, factor_parity=parity,
                price_tolerance_pct=tol, max_gap_sessions=gap,
                return_sessions=horizon, recent_days=recent)
    result = None; error = None; match_count = None
    # No computation on initial load; only user-requested screening.
    if request.args.get('run') == '1':
        if fmax - fmin > 49:
            error = 'اختاري نطاقًا لا يزيد عن 50 رقمًا في المرة الواحدة.'
        elif not _rate_limit(('search',_public_ip()), SEARCH_LIMIT_PER_MINUTE):
            error = 'طلبات كثيرة. انتظري دقيقة ثم أعيدي المحاولة.'
        else:
            cache_key = tuple(sorted(args.items(), key=lambda pair: pair[0]))
            try:
                with _cache_lock:
                    cached = _cache.get(cache_key)
                    if cached and time.monotonic()-cached[0] < CACHE_TTL_SECONDS:
                        raw_result = cached[1]
                    else:
                        raw_result = None
                if raw_result is None:
                    from monitor.seven_system import repeat_price_time_screener
                    raw_result = repeat_price_time_screener(**args)
                    with _cache_lock:
                        _cache.clear()
                        _cache[cache_key]=(time.monotonic(),raw_result)
                match_count = sum(bool(row.get('breakout_rule_match')) for row in raw_result['rows'])
                shown_rows = [row for row in raw_result['rows'] if row.get('breakout_rule_match')] if rule_only else raw_result['rows']
                result = dict(raw_result, rows=shown_rows, count=len(shown_rows))
            except Exception:
                app.logger.exception('Public screening failed')
                error = 'تعذر تحميل النتائج حاليًا. جربي مرة أخرى لاحقًا.'

    return render_template('public_screening.html', result=result, error=error,
                           factor_min=fmin,system_number=fmax,factor_parity=parity,
                           price_tolerance_pct=tol,max_gap_sessions=gap,
                           return_sessions=horizon,recent_days=recent if recent is not None else 'all',
                           rule_only=rule_only, rule_match_count=match_count,
                           refresh_message=request.args.get('update',''))

@app.post('/refresh')
def refresh():
    # Shared PostgreSQL cooldown. Atomic and race-safe across requests/restarts/services.
    try:
        DB = database()
        with DB.begin() as s:
            s.execute(text('''CREATE TABLE IF NOT EXISTS public_screening_refresh_gate (
                id integer PRIMARY KEY, next_allowed_at timestamptz NOT NULL,
                last_requested_at timestamptz NOT NULL
            )'''))
            s.execute(text("""INSERT INTO public_screening_refresh_gate(id,next_allowed_at,last_requested_at)
                VALUES(1, to_timestamp(0), to_timestamp(0)) ON CONFLICT(id) DO NOTHING"""))
            acquired = s.execute(text("""UPDATE public_screening_refresh_gate
                SET last_requested_at = NOW(), next_allowed_at = NOW() + interval '30 minutes'
                WHERE id=1 AND next_allowed_at <= NOW()
                RETURNING id""")).first() is not None
        if not acquired:
            return redirect(url_for('screening', update='wait'), code=303)
        # Launch the pre-existing updater. Separate process so web request stays responsive.
        subprocess.Popen([sys.executable,'-m','egx_public_refresh_once'],
                         cwd=os.getcwd(),stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,start_new_session=True)
        with _cache_lock: _cache.clear()
        return redirect(url_for('screening', update='started'), code=303)
    except Exception:
        app.logger.exception('Public refresh failed')
        return redirect(url_for('screening',update='failed'),code=303)

@app.errorhandler(404)
def missing(_):
    return Response('Not found', status=404, mimetype='text/plain')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT','8080')))
