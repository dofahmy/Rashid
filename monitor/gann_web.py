
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

    W,H=1200,580
    L,R,T,B=70,55,35,70
    iw,ih=W-L-R,H-T-B
    highs=[float(x) for x in C["high"]]
    lows=[float(x) for x in C["low"]]
    opens=[float(x) for x in C["open"]]
    closes=[float(x) for x in C["close"]]
    dates=C["dates"]

    main_future=a.get("tops",[])+a.get("lows",[])
    hist=a.get("previous_tops",[])+a.get("previous_lows",[])
    extra=[float(x["price"]) for x in main_future+hist if x.get("price")]
    actual=[float(x["actual_price"]) for x in hist if x.get("actual_price")]
    all_extra=extra+actual
    ymin=min(lows+all_extra) if all_extra else min(lows)
    ymax=max(highs+all_extra) if all_extra else max(highs)
    pad=max((ymax-ymin)*.08,max(ymax,1)*.01)
    ymin=max(0,ymin-pad); ymax=ymax+pad
    yr=max(ymax-ymin,1e-9)

    n=len(dates)
    def x(i): return L+(i/max(1,n-1))*iw
    def y(p): return T+(ymax-float(p))/yr*ih
    def esc(s):
        return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gann chart" style="width:100%;height:auto;background:#fff;border-radius:10px">']

    for k in range(6):
        p=ymin+(ymax-ymin)*k/5
        yy=y(p)
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{L-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{p:.2f}</text>')

    ticks=min(7,n)
    for k in range(ticks):
        i=round(k*(n-1)/max(1,ticks-1))
        xx=x(i)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H-B}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{xx:.1f}" y="{H-B+26}" text-anchor="middle" font-size="11" fill="#64748b">{esc(dates[i])}</text>')

    bw=max(1.5,min(6,iw/max(n,1)*.55))
    for i,(o,h,l,c) in enumerate(zip(opens,highs,lows,closes)):
        xx=x(i); col="#15803d" if c>=o else "#b91c1c"
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}" stroke="{col}" stroke-width="1"/>')
        yo,yc=y(o),y(c); top=min(yo,yc); height=max(1,abs(yc-yo))
        parts.append(f'<rect x="{xx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{height:.1f}" fill="{col}" opacity=".82"/>')

    import bisect
    def date_index(ds):
        ds=str(ds)
        pos=bisect.bisect_left(dates,ds)
        if pos<=0:return 0
        if pos>=n:return n-1
        d0=abs((__import__("datetime").date.fromisoformat(dates[pos-1])-__import__("datetime").date.fromisoformat(ds)).days)
        d1=abs((__import__("datetime").date.fromisoformat(dates[pos])-__import__("datetime").date.fromisoformat(ds)).days)
        return pos-1 if d0<=d1 else pos

    # Main future levels only: four dashed decision lines, labels at right edge.
    for j,it in enumerate(a.get("tops",[]),1):
        yy=y(it["price"]); col="#b91c1c"
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="6,5" opacity=".75"/>')
        parts.append(f'<polygon points="{W-R-6},{yy:.1f} {W-R-18},{yy-7:.1f} {W-R-18},{yy+7:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{W-R-24}" y="{yy-8:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{col}">قمة {j} · {float(it["price"]):.2f} · قرار {it["strength"]}%</text>')

    for j,it in enumerate(a.get("lows",[]),1):
        yy=y(it["price"]); col="#15803d"
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="6,5" opacity=".75"/>')
        parts.append(f'<polygon points="{W-R-6},{yy:.1f} {W-R-18},{yy-7:.1f} {W-R-18},{yy+7:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{W-R-24}" y="{yy+18:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{col}">قاع {j} · {float(it["price"]):.2f} · قرار {it["strength"]}%</text>')

    # Historical forecasts: arrows only, no full horizontal lines.
    def hist_arrow(it,is_top,index):
        ii=date_index(it.get("actual_date") or it.get("date"))
        xx=x(ii)
        # Put the marker on the realized extreme when available; otherwise forecast price.
        p=float(it.get("actual_price") or it["price"])
        yy=y(p)
        col=it.get("status_color","#64748b")
        if is_top:
            # Downward triangle positioned above the top.
            ay=max(T+10,yy-20)
            parts.append(f'<polygon points="{xx:.1f},{ay+12:.1f} {xx-8:.1f},{ay:.1f} {xx+8:.1f},{ay:.1f}" fill="{col}"/>')
            ty=max(T+10,ay-5)
            label=f'قمة سابقة {index} · {it.get("status","")}'
            parts.append(f'<text x="{xx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="{col}">{esc(label)}</text>')
        else:
            # Upward triangle positioned below the low.
            ay=min(H-B-10,yy+20)
            parts.append(f'<polygon points="{xx:.1f},{ay-12:.1f} {xx-8:.1f},{ay:.1f} {xx+8:.1f},{ay:.1f}" fill="{col}"/>')
            ty=min(H-B-2,ay+14)
            label=f'قاع سابق {index} · {it.get("status","")}'
            parts.append(f'<text x="{xx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="10" font-weight="700" fill="{col}">{esc(label)}</text>')

    for j,it in enumerate(a.get("previous_tops",[]),1):
        hist_arrow(it,True,j)
    for j,it in enumerate(a.get("previous_lows",[]),1):
        hist_arrow(it,False,j)

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
