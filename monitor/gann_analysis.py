
from __future__ import annotations
import math
from datetime import date, datetime, timedelta
from typing import Any
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select, func

IMPORTANT_RATIOS=[(0.25,"1/4",82),(1/3,"1/3",78),(0.375,"3/8",72),(0.50,"1/2",92),(0.625,"5/8",72),(2/3,"2/3",78),(0.75,"3/4",82),(0.875,"7/8",72),(1.00,"1/1",95)]
SQ9_ANGLES=[(45,78),(90,90),(135,78),(180,94),(225,78),(270,90),(315,78),(360,96)]
MASTER_144=[(36,86),(45,80),(48,78),(54,80),(63,82),(72,94),(81,82),(90,90),(96,78),(108,86),(117,76),(126,78),(135,82),(144,98)]
MIKULA_225_CELL_COUNTS=[9,25,49,81,121,169,225,289,361]

def _business_add(d,n): return (pd.Timestamp(d)+pd.offsets.BDay(max(0,int(n)))).date()
def _calendar_add(d,n): return d+timedelta(days=max(0,int(round(n))))
def _safe_float(v,default=None):
    try:
        x=float(v); return x if math.isfinite(x) else default
    except Exception: return default

def _daily_table(session):
    return Table("market_candles_1d",MetaData(),autoload_with=session.get_bind())

def load_daily(DB,symbol,limit=900):
    symbol=(symbol or "").strip().upper()
    with DB() as s:
        t=_daily_table(s)
        rows=s.execute(select(t.c.session_date,t.c.o,t.c.h,t.c.l,t.c.c,t.c.v,t.c.adj_c).where(t.c.symbol==symbol).order_by(t.c.session_date.desc()).limit(limit)).all()
    if not rows: return pd.DataFrame(columns=["d","o","h","l","c","v","adj_c"])
    df=pd.DataFrame(rows,columns=["d","o","h","l","c","v","adj_c"]).iloc[::-1].reset_index(drop=True)
    df["d"]=pd.to_datetime(df["d"]).dt.date
    for c in ["o","h","l","c","v","adj_c"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["h","l","c"]).reset_index(drop=True)

def normalize_market(market):
    market=(market or "US").strip().upper()
    if market in ("EGX","EGYPT","CA","مصر"):
        return "EGX"
    if market in ("ALL","كل","الكل"):
        return "ALL"
    return "US"

def symbol_market(symbol):
    s=(symbol or "").strip().upper()
    return "EGX" if s.endswith(".CA") else "US"

def list_symbols(DB,query="",page=1,per_page=100,market="US"):
    query=(query or "").strip().upper()[:40]
    market=normalize_market(market)
    page=max(1,int(page or 1))
    per_page=max(20,min(200,int(per_page or 100)))
    with DB() as s:
        t=_daily_table(s)
        cond=[]
        if query:
            cond.append(t.c.symbol.ilike(f"%{query}%"))
        # Yahoo Finance Egypt symbols are stored with .CA suffix.
        if market=="EGX":
            cond.append(t.c.symbol.ilike("%.CA"))
        elif market=="US":
            cond.append(~t.c.symbol.ilike("%.CA"))
        sq=select(t.c.symbol).where(*cond).distinct().subquery()
        total=int(s.scalar(select(func.count()).select_from(sq)) or 0)
        symbols=list(
            s.scalars(
                select(sq.c.symbol)
                .order_by(sq.c.symbol)
                .offset((page-1)*per_page)
                .limit(per_page)
            ).all()
        )
    pages=max(1,math.ceil(total/per_page))
    return symbols,total,min(page,pages),pages

def _atr(df,n=14):
    if len(df)<2:return 0.0
    prev=df["c"].shift(1)
    tr=pd.concat([(df["h"]-df["l"]).abs(),(df["h"]-prev).abs(),(df["l"]-prev).abs()],axis=1).max(axis=1)
    x=float(tr.tail(n).mean()); return x if math.isfinite(x) else 0.0

def detect_pivots(df,wing=7,min_move_pct=3.0):
    if len(df)<wing*2+3:return []
    h=df["h"].to_numpy(float); l=df["l"].to_numpy(float); piv=[]
    for i in range(wing,len(df)-wing):
        if h[i]>=np.nanmax(h[i-wing:i+wing+1]): piv.append({"i":i,"kind":"H","price":float(h[i]),"date":df.at[i,"d"]})
        if l[i]<=np.nanmin(l[i-wing:i+wing+1]): piv.append({"i":i,"kind":"L","price":float(l[i]),"date":df.at[i,"d"]})
    piv.sort(key=lambda x:(x["i"],0 if x["kind"]=="L" else 1))
    clean=[]
    for p in piv:
        if clean and p["kind"]==clean[-1]["kind"]:
            better=(p["kind"]=="H" and p["price"]>=clean[-1]["price"]) or (p["kind"]=="L" and p["price"]<=clean[-1]["price"])
            if better: clean[-1]=p
            continue
        clean.append(p)
    out=[]
    for p in clean:
        if not out: out.append(p); continue
        prev=out[-1]
        move=abs(p["price"]/prev["price"]-1)*100 if prev["price"] else 0
        if move>=min_move_pct: out.append(p)
    return out[-20:]

def _latest_pair(p):
    lows=[x for x in p if x["kind"]=="L"]; highs=[x for x in p if x["kind"]=="H"]
    return (lows[-1] if lows else None),(highs[-1] if highs else None)

def _last_completed_range(p):
    if len(p)<2:return None
    a,b=p[-2],p[-1]
    return {"start":a,"end":b,"low":min(a["price"],b["price"]),"high":max(a["price"],b["price"]),"range":abs(b["price"]-a["price"]),"anchor_date":b["date"]}

def _sq9_levels(pivot_price,pivot_kind,pivot_date):
    root=math.sqrt(max(pivot_price,1e-12)); out=[]
    for angle,strength in SQ9_ANGLES:
        d=angle/180.0; up=(root+d)**2; down=(root-d)**2 if root>d else None
        out.append({"method":"Mikula SQ9","submethod":f"{angle}° أعلى","level":up,"strength":strength,"pivot_kind":pivot_kind,"pivot_date":pivot_date})
        if down and down>0: out.append({"method":"Mikula SQ9","submethod":f"{angle}° أسفل","level":down,"strength":strength,"pivot_kind":pivot_kind,"pivot_date":pivot_date})
    return out

def _range_ratio_levels(r):
    if not r or r["range"]<=0:return []
    lo,hi=r["low"],r["high"]; out=[]
    for frac,label,strength in IMPORTANT_RATIOS:
        out.append({"method":"Gann Range Ratios","submethod":label,"level":lo+(hi-lo)*frac,"strength":strength,"pivot_kind":"RANGE","pivot_date":r["anchor_date"]})
    return out

def _master_price_levels(p):
    if not p:return []
    out=[]; x=float(p["price"])
    for n,strength in MASTER_144:
        out.append({"method":"Master 144","submethod":f"+{n} نقطة","level":x+n,"strength":strength,"pivot_kind":p["kind"],"pivot_date":p["date"]})
        if x-n>0: out.append({"method":"Master 144","submethod":f"-{n} نقطة","level":x-n,"strength":strength,"pivot_kind":p["kind"],"pivot_date":p["date"]})
    return out

def cluster_price_levels(items,current,atr):
    items=sorted([x for x in items if _safe_float(x.get("level")) and x["level"]>0],key=lambda x:x["level"])
    if not items:return []
    tol=max(current*0.004,atr*0.25 if atr else 0); cs=[]
    for it in items:
        if not cs or abs(it["level"]-cs[-1]["level"])>tol: cs.append({"items":[it],"level":it["level"]})
        else:
            c=cs[-1]; c["items"].append(it)
            w=np.array([max(1,i["strength"]) for i in c["items"]]); v=np.array([i["level"] for i in c["items"]]); c["level"]=float(np.average(v,weights=w))
    out=[]
    for c in cs:
        methods=sorted(set(i["method"] for i in c["items"])); base=max(i["strength"] for i in c["items"])
        bonus=min(18,7*(len(methods)-1)+3*(len(c["items"])-len(methods)))
        st=min(100,int(round(base+bonus)))
        out.append({"level":round(c["level"],4),"strength":st,"methods":methods,"labels":[i["submethod"] for i in c["items"]],"side":"resistance" if c["level"]>current else "support","distance_pct":round(100*(c["level"]/current-1),2),"count":len(c["items"])})
    return out

def _time_strength(label):
    return {"1/1":95,"1/2":92,"1/4":82,"3/4":82,"1/3":78,"2/3":78,"3/8":72,"5/8":72,"7/8":72}.get(label,75)

def _repeat_square_dates(anchor,unit,method,max_cycles=9):
    if not unit or unit<=0 or unit>5000:return []
    today=date.today(); out=[]
    for cycle in range(max_cycles):
        for frac,label,_ in IMPORTANT_RATIOS:
            off=(cycle+frac)*unit
            if off>6000: continue
            for mode in ("calendar","trading"):
                d=_calendar_add(anchor,off) if mode=="calendar" else _business_add(anchor,int(round(off)))
                if d>=today-timedelta(days=8): out.append({"method":method,"submethod":f"{label} · {'تقويمي' if mode=='calendar' else 'جلسات'}","date":d,"strength":_time_strength(label),"mode":mode})
    return out

def _master_144_dates(anchor):
    out=[]; today=date.today()
    for cycle in range(8):
        for n,strength in MASTER_144:
            off=cycle*144+n
            for mode in ("calendar","trading"):
                d=_calendar_add(anchor,off) if mode=="calendar" else _business_add(anchor,off)
                if d>=today-timedelta(days=8): out.append({"method":"Master 144","submethod":f"{n} · {'تقويمي' if mode=='calendar' else 'جلسات'}","date":d,"strength":strength,"mode":mode})
    return out

def _mikula_dates(anchor):
    out=[]; today=date.today()
    for n in MIKULA_225_CELL_COUNTS:
        d=_business_add(anchor,n)
        if d>=today-timedelta(days=8): out.append({"method":"Mikula 225° Cells","submethod":f"{n} bars","date":d,"strength":min(96,78+int(math.log(max(n,9),3))*3),"mode":"trading"})
    return out

def cluster_time_dates(items,window_days=2):
    if not items:return []
    items=sorted(items,key=lambda x:x["date"]); cs=[]
    for it in items:
        if not cs or abs((it["date"]-cs[-1]["date"]).days)>window_days: cs.append({"date":it["date"],"items":[it]})
        else:
            cs[-1]["items"].append(it); cs[-1]["date"]=max(cs[-1]["items"],key=lambda x:x["strength"])["date"]
    out=[]
    for c in cs:
        methods=sorted(set(i["method"] for i in c["items"])); base=max(i["strength"] for i in c["items"])
        st=min(100,int(round(base+min(18,8*(len(methods)-1)+2*(len(c["items"])-len(methods))))))
        out.append({"date":c["date"],"strength":st,"methods":methods,"labels":[i["submethod"] for i in c["items"]],"count":len(c["items"])})
    return out

def strength_color(v):
    v=int(v or 0)
    return "#b91c1c" if v>=90 else "#ea580c" if v>=80 else "#ca8a04" if v>=70 else "#2563eb" if v>=60 else "#64748b"


def _historical_expected_turns(df, pivots, chart_start):
    """
    Build the latest historical Gann forecasts using information available
    from each completed pivot anchor.

    Low pivot  -> expected TOP using upward SQ9/Master price levels + future time cluster.
    High pivot -> expected LOW using downward SQ9/Master price levels + future time cluster.

    These are forecast markers, not labels taken from future realized highs/lows.
    """
    if not pivots:
        return [], []

    today = date.today()
    hist_tops = []
    hist_lows = []

    for idx, p in enumerate(pivots[-12:]):
        anchor_date = p["date"]
        anchor_price = float(p["price"])

        # Price projections known at the pivot.
        price_items = _sq9_levels(anchor_price, p["kind"], anchor_date) + _master_price_levels(p)
        if idx > 0:
            prev = pivots[max(0, len(pivots)-12)+idx-1] if len(pivots) > 12 else pivots[idx-1]
            if prev:
                rng = {
                    "low": min(float(prev["price"]), anchor_price),
                    "high": max(float(prev["price"]), anchor_price),
                    "range": abs(anchor_price-float(prev["price"])),
                    "anchor_date": anchor_date,
                }
                price_items += _range_ratio_levels(rng)

        # Time projections known at the pivot. Here we intentionally keep past dates too.
        time_items = []
        unit = anchor_price
        if unit > 0 and unit <= 5000:
            for cycle in range(0, 5):
                for frac, label, _ in IMPORTANT_RATIOS:
                    off = (cycle + frac) * unit
                    if off <= 0 or off > 2500:
                        continue
                    for mode in ("calendar", "trading"):
                        d = _calendar_add(anchor_date, off) if mode=="calendar" else _business_add(anchor_date, int(round(off)))
                        if d > anchor_date and d < today and d >= chart_start:
                            time_items.append({
                                "method": "Gann Square Low" if p["kind"]=="L" else "Gann Square High",
                                "submethod": f"{label} · {'تقويمي' if mode=='calendar' else 'جلسات'}",
                                "date": d,
                                "strength": _time_strength(label),
                                "mode": mode,
                            })

        for cycle in range(0, 5):
            for n, strength in MASTER_144:
                off = cycle*144+n
                for mode in ("calendar", "trading"):
                    d = _calendar_add(anchor_date, off) if mode=="calendar" else _business_add(anchor_date, off)
                    if d > anchor_date and d < today and d >= chart_start:
                        time_items.append({
                            "method":"Master 144",
                            "submethod":f"{n} · {'تقويمي' if mode=='calendar' else 'جلسات'}",
                            "date":d,
                            "strength":strength,
                            "mode":mode,
                        })

        for n in MIKULA_225_CELL_COUNTS:
            d = _business_add(anchor_date, n)
            if d > anchor_date and d < today and d >= chart_start:
                time_items.append({
                    "method":"Mikula 225° Cells",
                    "submethod":f"{n} bars",
                    "date":d,
                    "strength":min(96,78+int(math.log(max(n,9),3))*3),
                    "mode":"trading",
                })

        time_clusters = cluster_time_dates(time_items)
        if not time_clusters:
            continue

        # Use the first materially strong time cluster after the anchor.
        strong_times = [x for x in time_clusters if x["strength"] >= 78]
        if not strong_times:
            continue

        if p["kind"] == "L":
            candidates = [x for x in price_items if x["level"] > anchor_price]
            if not candidates:
                continue
            candidates.sort(key=lambda x:(x["level"]-anchor_price, -x["strength"]))
            price_pick = candidates[0]
            for t in strong_times[:2]:
                st = int(round(min(100, .62*price_pick["strength"] + .38*t["strength"] + 5)))
                hist_tops.append({
                    "type":"TOP",
                    "price":round(float(price_pick["level"]),4),
                    "date":t["date"],
                    "strength":st,
                    "methods":sorted(set([price_pick["method"]] + t["methods"])),
                    "color":strength_color(st),
                    "anchor_date":anchor_date,
                    "anchor_price":round(anchor_price,4),
                })
        else:
            candidates = [x for x in price_items if 0 < x["level"] < anchor_price]
            if not candidates:
                continue
            candidates.sort(key=lambda x:(anchor_price-x["level"], -x["strength"]))
            price_pick = candidates[0]
            for t in strong_times[:2]:
                st = int(round(min(100, .62*price_pick["strength"] + .38*t["strength"] + 5)))
                hist_lows.append({
                    "type":"LOW",
                    "price":round(float(price_pick["level"]),4),
                    "date":t["date"],
                    "strength":st,
                    "methods":sorted(set([price_pick["method"]] + t["methods"])),
                    "color":strength_color(st),
                    "anchor_date":anchor_date,
                    "anchor_price":round(anchor_price,4),
                })

    # Deduplicate close dates and keep the strongest forecast.
    def compact(items):
        items = sorted(items, key=lambda x:(x["date"], -x["strength"]))
        out=[]
        for it in items:
            if out and abs((it["date"]-out[-1]["date"]).days) <= 2:
                if it["strength"] > out[-1]["strength"]:
                    out[-1]=it
            else:
                out.append(it)
        return out[-2:]

    return compact(hist_tops), compact(hist_lows)

def _pair_turns(pc,tc,current):
    sups=sorted([x for x in pc if x["level"]<current],key=lambda x:(abs(x["distance_pct"]),-x["strength"]))[:2]
    ress=sorted([x for x in pc if x["level"]>current],key=lambda x:(abs(x["distance_pct"]),-x["strength"]))[:2]
    future=sorted([x for x in tc if x["date"]>=date.today()],key=lambda x:x["date"])[:4]
    def make(levels,typ):
        ans=[]
        for i,p in enumerate(levels):
            t=future[i] if i<len(future) else (future[-1] if future else None)
            st=int(round(p["strength"] if not t else min(100,.62*p["strength"]+.38*t["strength"]+5)))
            ans.append({"type":typ,"price":p["level"],"date":t["date"] if t else None,"strength":st,"price_strength":p["strength"],"time_strength":t["strength"] if t else None,"methods":sorted(set(p["methods"]+(t["methods"] if t else []))),"color":strength_color(st)})
        return ans
    return make(ress,"TOP"),make(sups,"LOW")

def build_method_rows(last,pc,tc):
    methods=[
      ("Gann Square Low","سعر القاع = الزمن مع الكسور المهمة."),
      ("Gann Square High","سعر القمة = الزمن مع الكسور المهمة."),
      ("Gann Square Range","مدى آخر حركة = الزمن ويتكرر طالما المدى صالح."),
      ("Master 144","نقاط 36/45/48/54/63/72/81/90/96/108/117/126/135/144."),
      ("Mikula SQ9","مستويات السعر من الجذر التربيعي عند 45° حتى 360°."),
      ("Mikula 225° Cells","تواريخ bars: 9،25،49،81،121…"),
      ("Gann Range Ratios","1/4،1/3،3/8،1/2،5/8،2/3،3/4،7/8 داخل آخر Range.")]
    rows=[]
    for method,desc in methods:
        xs=[x for x in pc if method in x["methods"]]
        sup=max([x for x in xs if x["level"]<last],key=lambda x:x["level"],default=None)
        res=min([x for x in xs if x["level"]>last],key=lambda x:x["level"],default=None)
        times=[x for x in tc if method in x["methods"] and x["date"]>=date.today()][:2]
        sts=[x["strength"] for x in [sup,res] if x]+[x["strength"] for x in times]
        st=max(sts) if sts else 0
        rows.append({"method":method,"description":desc,"support":sup,"resistance":res,"times":times,"strength":st,"color":strength_color(st)})
    return rows

def analyze_symbol(DB,symbol,limit=900):
    df=load_daily(DB,symbol,limit)
    if df.empty:return None
    piv=detect_pivots(df); low,high=_latest_pair(piv); rng=_last_completed_range(piv); last=float(df.iloc[-1]["c"]); atr=_atr(df)
    pi=[]
    if low: pi+=_sq9_levels(low["price"],"L",low["date"])+_master_price_levels(low)
    if high: pi+=_sq9_levels(high["price"],"H",high["date"])+_master_price_levels(high)
    pi+=_range_ratio_levels(rng); pc=cluster_price_levels(pi,last,atr)
    ti=[]
    if low: ti+=_repeat_square_dates(low["date"],low["price"],"Gann Square Low")+_master_144_dates(low["date"])+_mikula_dates(low["date"])
    if high: ti+=_repeat_square_dates(high["date"],high["price"],"Gann Square High")+_master_144_dates(high["date"])+_mikula_dates(high["date"])
    if rng and rng["range"]>0: ti+=_repeat_square_dates(rng["anchor_date"],rng["range"],"Gann Square Range")
    tc=cluster_time_dates(ti); next_times=[x for x in tc if x["date"]>=date.today()][:6]
    tops,lows=_pair_turns(pc,tc,last)
    ns=max([x for x in pc if x["level"]<last],key=lambda x:x["level"],default=None)
    nr=min([x for x in pc if x["level"]>last],key=lambda x:x["level"],default=None)
    strengths=[x["strength"] for x in tops+lows]+[x["strength"] for x in next_times[:2]]
    candles=df.tail(220)
    chart_start=candles.iloc[0]["d"] if len(candles) else df.iloc[0]["d"]
    previous_tops,previous_lows=_historical_expected_turns(df,piv,chart_start)
    return {"symbol":symbol.upper(),"last_price":round(last,4),"last_date":df.iloc[-1]["d"],"atr14":round(atr,4),"latest_low":low,"latest_high":high,"range":rng,"price_clusters":pc,"time_clusters":tc,"next_times":next_times,"nearest_support":ns,"nearest_resistance":nr,"tops":tops,"lows":lows,"previous_tops":previous_tops,"previous_lows":previous_lows,"overall_strength":max(strengths) if strengths else 0,"overall_color":strength_color(max(strengths) if strengths else 0),"chart":{"dates":[d.isoformat() for d in candles["d"]],"open":[round(float(x),4) for x in candles["o"]],"high":[round(float(x),4) for x in candles["h"]],"low":[round(float(x),4) for x in candles["l"]],"close":[round(float(x),4) for x in candles["c"]]},"method_rows":build_method_rows(last,pc,tc)}

def market_page(DB,query="",page=1,per_page=100,market="US"):
    market=normalize_market(market)
    symbols,total,page,pages=list_symbols(DB,query,page,per_page,market=market); rows=[]
    for sym in symbols:
        try:
            a=analyze_symbol(DB,sym,420)
            if not a: continue
            rows.append({"symbol":sym,"last_price":a["last_price"],"last_date":a["last_date"],"support":a["nearest_support"],"resistance":a["nearest_resistance"],"next_time":a["next_times"][0] if a["next_times"] else None,"top1":a["tops"][0] if a["tops"] else None,"top2":a["tops"][1] if len(a["tops"])>1 else None,"low1":a["lows"][0] if a["lows"] else None,"low2":a["lows"][1] if len(a["lows"])>1 else None,"strength":a["overall_strength"],"color":a["overall_color"]})
        except Exception as e:
            rows.append({"symbol":sym,"error":str(e)[:180],"strength":0,"color":"#64748b"})
    return {"rows":rows,"total":total,"page":page,"pages":pages,"query":query or "","market":market}

def source_methodology():
    return [
      {"name":"Gann: Square Range / Low / High","rule":"مساواة عدد نقاط السعر بعدد فترات الزمن مع نسب 1/4،1/3،1/2،2/3،3/4 والمربع الكامل.","source":"W.D. Gann Master Commodities Course — Chapter 6."},
      {"name":"Gann Master Square of 144","rule":"نقاط وتقاطع 36،45،48،54،63،72،81،90،96،108،117،126،135،144.","source":"W.D. Gann Master Mathematical Price, Time and Trend Calculator."},
      {"name":"Bowden","rule":"Squaring a Low / High / Range مع calendar days وtrading days وضبط 45° كـ1×1 حقيقي.","source":"David E. Bowden — Squaring Time and Price."},
      {"name":"Mikula Square of Nine","rule":"(sqrt(P) ± angle/180)^2 ومستويات Cardinal/Fixed Cross وتواريخ Cell counts.","source":"Patrick Mikula — The Definitive Guide to Forecasting Using W.D. Gann's Square of Nine."}
    ]
