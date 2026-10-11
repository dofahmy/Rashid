"""Password-protected browser export for the Egyptian Pre-Burst V7.6 report.

Integrate in the existing Flask app: app.register_blueprint(egypt_export_bp)
Does not change trading strategy, historical ledgers, or app.py automatically.
"""
from __future__ import annotations

import hmac
import os
import threading
import traceback
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from flask import Blueprint, Response, request, send_file


egypt_export_bp = Blueprint('preburst_egypt_browser_export', __name__)
_DIR = Path(os.environ.get('PREBURST_EGYPT_EXPORT_DIR', '/tmp/preburst-egypt-web-export'))
_ZIP = _DIR / 'egypt_preburst_results.zip'
_CSV = _DIR / 'nonoverlapping_cycles_250_trading_days.csv'
_SUMMARY = _DIR / 'summary.json'
_LOCK = threading.Lock()
_STATE = {'running': False, 'error': None, 'finished': None}


def _protect():
    password = os.environ.get('PREBURST_EXPORT_PASSWORD', '')
    if not password or len(password) < 12:
        return Response('Set PREBURST_EXPORT_PASSWORD in Railway Web Variables (at least 12 characters).', status=503)
    auth = request.authorization
    if not auth or not (hmac.compare_digest(auth.username or '', 'admin') and hmac.compare_digest(auth.password or '', password)):
        return Response('Authentication required', status=401, headers={'WWW-Authenticate': 'Basic realm="PreBurst Egypt Export", charset="UTF-8"'})
    return None


def _html(body: str, status=200):
    result = Response('<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>تصدير Pre-Burst مصر</title><style>body{font-family:system-ui,sans-serif;max-width:670px;margin:55px auto;padding:18px;line-height:1.9;background:#f7f9fa}main{background:white;padding:28px;border:1px solid #ddd;border-radius:16px}button,a.button{background:#146347;color:white;border:0;border-radius:9px;padding:13px 18px;font-size:17px;cursor:pointer;text-decoration:none}small{color:#666}</style><main><h2>تصدير نتائج Pre-Burst — مصر</h2>' + body + '</main></html>', status=status, mimetype='text/html')
    result.headers['Cache-Control'] = 'no-store'
    result.headers['X-Content-Type-Options'] = 'nosniff'
    return result


def _build():
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        from monitor.us_preburst_250day_outcomes import run
        summary = run('egypt', str(_DIR), '')
        if not _CSV.is_file() or not _SUMMARY.is_file():
            raise FileNotFoundError('V7.6 did not create both the CSV and summary.json')
        temp = _DIR / 'egypt_preburst_results.building.zip'
        with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
            z.write(_CSV, arcname=_CSV.name)
            z.write(_SUMMARY, arcname=_SUMMARY.name)
        temp.replace(_ZIP)
        with _LOCK:
            _STATE['finished'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            _STATE['error'] = None
    except Exception:
        with _LOCK:
            _STATE['error'] = traceback.format_exc()[-2000:]
    finally:
        with _LOCK:
            _STATE['running'] = False


@egypt_export_bp.route('/preburst/egypt-export', methods=['GET', 'POST'])
def page():
    denied = _protect()
    if denied is not None:
        return denied
    if request.method == 'POST':
        with _LOCK:
            if not _STATE['running']:
                _STATE['running'] = True
                _STATE['error'] = None
                threading.Thread(target=_build, name='preburst-egypt-export', daemon=True).start()
    with _LOCK:
        state = dict(_STATE)
    if state['running']:
        return _html('<p>جاري بناء التقرير من قاعدة البيانات في خدمة Web. انتظري ثم حدّثي الصفحة.</p><meta http-equiv="refresh" content="8">')
    if state['error']:
        return _html('<p>فشل بناء التقرير. آخر تفاصيل الخطأ:</p><pre style="white-space:pre-wrap;direction:ltr;text-align:left">'+escape(state['error'])+'</pre><form method="post"><button>إعادة المحاولة</button></form>', 500)
    if _ZIP.exists():
        size_kb = round(_ZIP.stat().st_size / 1024, 1)
        when = state['finished'] or 'تقرير محفوظ داخل خدمة Web'
        return _html(f'<p>التقرير جاهز — {size_kb} KB</p><p><small>آخر إنشاء: {escape(when)}</small></p><p><a class="button" href="/preburst/egypt-export/download">تحميل التقرير ZIP</a></p><form method="post"><button>تحديث التقرير من قاعدة البيانات</button></form><p><small>الملف يحتوي CSV لكل دورات مصر وsummary.json.</small></p>')
    return _html('<p>هيتم إنشاء ملف النتائج داخل Web من قاعدة PostgreSQL. لا تحتاجي ملف Railway Console المؤقت.</p><form method="post"><button>إنشاء التقرير المصري</button></form>')


@egypt_export_bp.route('/preburst/egypt-export/download', methods=['GET'])
def download():
    denied = _protect()
    if denied is not None:
        return denied
    if not _ZIP.is_file():
        return _html('<p>التقرير غير متاح في حاوية Web الحالية. افتحي صفحة التصدير واضغطي إنشاء التقرير.</p>', 404)
    response = send_file(_ZIP, as_attachment=True, download_name='egypt_preburst_results.zip', mimetype='application/zip', max_age=0)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response
