
from __future__ import annotations

from flask import Blueprint, render_template, request, session, redirect, url_for, abort
from markupsafe import Markup
from core import database
from monitor.gann_analysis import market_page, source_methodology, analyze_symbol

gann_bp = Blueprint("gann", __name__)
DB = database()

def _require_admin():
    if not session.get("admin"):
        return redirect(url_for("login"))
    return None

def _svg_chart(a):
    C=a["chart"]
    if not C["dates"]:
        return '<div class="muted">لا توجد بيانات كافية للشارت.</div>'

    W,H=1200,560
    L,R,T,B=70,30,30,65
    iw,ih=W-L-R,H-T-B
    highs=[float(x) for x in C["high"]]
    lows=[float(x) for x in C["low"]]
    opens=[float(x) for x in C["open"]]
    closes=[float(x) for x in C["close"]]
    dates=C["dates"]

    extra=[x["price"] for x in a.get("tops",[])+a.get("lows",[])+a.get("previous_tops",[])+a.get("previous_lows",[]) if x.get("price")]
    ymin=min(lows+extra) if extra else min(lows)
    ymax=max(highs+extra) if extra else max(highs)
    pad=max((ymax-ymin)*.08, max(ymax,1)*.01)
    ymin=max(0,ymin-pad); ymax=ymax+pad
    yr=max(ymax-ymin,1e-9)

    n=len(dates)
    def x(i): return L + (i/(max(1,n-1)))*iw
    def y(p): return T + (ymax-float(p))/yr*ih
    def esc(s): 
        return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gann chart" style="width:100%;height:auto;background:#fff;border-radius:10px">']
    # grid / labels
    for k in range(6):
        p=ymin+(ymax-ymin)*k/5
        yy=y(p)
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{L-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{p:.2f}</text>')
    # x labels
    ticks=min(7,n)
    for k in range(ticks):
        i=round(k*(n-1)/max(1,ticks-1))
        xx=x(i)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H-B}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{xx:.1f}" y="{H-B+25}" text-anchor="middle" font-size="11" fill="#64748b">{esc(dates[i])}</text>')

    # candles
    bw=max(1.5,min(6,iw/max(n,1)*.55))
    for i,(o,h,l,c) in enumerate(zip(opens,highs,lows,closes)):
        xx=x(i); col="#15803d" if c>=o else "#b91c1c"
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}" stroke="{col}" stroke-width="1"/>')
        yo,yc=y(o),y(c); top=min(yo,yc); height=max(1,abs(yc-yo))
        parts.append(f'<rect x="{xx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{height:.1f}" fill="{col}" opacity=".82"/>')

    # helper: date index nearest
    import bisect
    def date_index(ds):
        pos=bisect.bisect_left(dates,str(ds))
        if pos<=0:return 0
        if pos>=n:return n-1
        return pos if abs(pos-(n-1)/2)<abs((pos-1)-(n-1)/2) else pos-1

    def add_forecast(items,label,past=False):
        for j,it in enumerate(items,1):
            price=float(it["price"]); col=it.get("color","#2563eb")
            yy=y(price)
            dash="6,5" if past else "3,3"
            parts.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.5" stroke-dasharray="{dash}" opacity=".85"/>')
            if it.get("date"):
                ii=date_index(it["date"]); xx=x(ii)
                parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H-B}" stroke="{col}" stroke-width="1" stroke-dasharray="{dash}" opacity=".65"/>')
                boxx=max(L+5,min(W-R-170,xx+6)); boxy=max(T+18,min(H-B-8,yy-8))
                suffix=" سابقة" if past else ""
                txt=f'{label}{suffix} {j} · {price:.2f} · {it["strength"]}%'
                parts.append(f'<rect x="{boxx:.1f}" y="{boxy-16:.1f}" width="165" height="21" rx="4" fill="{col}" opacity=".90"/>')
                parts.append(f'<text x="{boxx+5:.1f}" y="{boxy-2:.1f}" font-size="11" fill="#fff">{esc(txt)}</text>')

    add_forecast(a.get("previous_tops",[]),"قمة",True)
    add_forecast(a.get("previous_lows",[]),"قاع",True)
    add_forecast(a.get("tops",[]),"قمة",False)
    add_forecast(a.get("lows",[]),"قاع",False)

    parts.append('</svg>')
    return "".join(parts)

@gann_bp.get("/gann")
def gann_analysis():
    r=_require_admin()
    if r:return r
    try: page=max(1,int(request.args.get("page","1")))
    except ValueError: page=1
    try: per_page=int(request.args.get("per_page","100"))
    except ValueError: per_page=100
    query=request.args.get("symbol","").strip().upper()[:40]
    market=request.args.get("market","US").strip().upper()
    if market not in ("US","EGX","ALL"):
        market="US"
    data=market_page(DB,query=query,page=page,per_page=per_page,market=market)
    return render_template(
        "gann_analysis.html",
        methodology=source_methodology(),
        per_page=max(20,min(200,per_page)),
        **data
    )

@gann_bp.get("/gann/<symbol>")
def gann_symbol(symbol):
    r=_require_admin()
    if r:return r
    symbol=(symbol or "").strip().upper()[:40]
    a=analyze_symbol(DB,symbol)
    if not a:abort(404)
    chart_svg=Markup(_svg_chart(a))
    return render_template("gann_symbol.html",a=a,chart_svg=chart_svg)
