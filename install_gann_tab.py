#!/usr/bin/env python3
from pathlib import Path
import shutil,re
ROOT=Path(__file__).resolve().parent
APP=ROOT/'app.py'; T=ROOT/'templates'
if not APP.exists(): raise SystemExit('app.py not found in project root')
app=APP.read_text(encoding='utf-8')
marker='# === GANN ANALYSIS TAB ==='
if marker not in app:
    injection = r'''
    # === GANN ANALYSIS TAB ===
    @app.get('/gann')
    @auth
    def gann_analysis():
        from monitor.gann_analysis import market_page, source_methodology
        from flask import request, render_template
        try: page=max(1,int(request.args.get('page','1')))
        except ValueError: page=1
        try: per_page=int(request.args.get('per_page','100'))
        except ValueError: per_page=100
        query=request.args.get('symbol','').strip().upper()[:40]
        data=market_page(DB,query=query,page=page,per_page=per_page)
        return render_template('gann_analysis.html',methodology=source_methodology(),per_page=max(20,min(200,per_page)),**data)

    @app.get('/gann/<symbol>')
    @auth
    def gann_symbol(symbol):
        from monitor.gann_analysis import analyze_symbol
        from flask import render_template, abort
        symbol=(symbol or '').strip().upper()[:40]
        a=analyze_symbol(DB,symbol)
        if not a: abort(404)
        return render_template('gann_symbol.html',a=a)
'''
    ms=list(re.finditer(r'(?m)^    return app\s*$',app))
    if not ms: raise SystemExit("Could not find final '    return app' in app.py")
    shutil.copy2(APP,APP.with_suffix('.py.before_gann'))
    pos=ms[-1].start(); APP.write_text(app[:pos]+injection+'\n'+app[pos:],encoding='utf-8')
    print('Patched app.py')
else: print('app.py already patched')
base=T/'base.html'
if base.exists():
    txt=base.read_text(encoding='utf-8')
    if 'gann_analysis' not in txt:
        shutil.copy2(base,base.with_suffix('.html.before_gann'))
        link='<a href="{{url_for(\'gann_analysis\')}}">تحليل جان</a>'
        m=re.search(r'(<a[^>]+url_for\([\'\"]stocks[\'\"]\)[^>]*>.*?</a>)',txt,flags=re.S)
        if m: txt=txt[:m.end()]+'\n'+link+txt[m.end():]
        elif '</nav>' in txt: txt=txt.replace('</nav>',link+'\n</nav>',1)
        elif '</header>' in txt: txt=txt.replace('</header>',link+'\n</header>',1)
        else: txt=link+'\n'+txt
        base.write_text(txt,encoding='utf-8'); print('Patched templates/base.html')
stocks=T/'stocks.html'
if stocks.exists():
    txt=stocks.read_text(encoding='utf-8')
    if 'gann_analysis' not in txt:
        shutil.copy2(stocks,stocks.with_suffix('.html.before_gann'))
        needle="{% block content %}"; add='\n<div class="actions"><a class="button secondary" href="{{url_for(\'gann_analysis\')}}">تحليل جان</a></div>\n'
        stocks.write_text(txt.replace(needle,needle+add,1) if needle in txt else add+txt,encoding='utf-8')
        print('Patched templates/stocks.html')
print('Done. Deploy Web and open /gann')
