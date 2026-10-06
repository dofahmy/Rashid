#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGX Sector / Market Turn-Window Map V1
======================================

Idea:
    We do NOT start from every planetary date.
    First find the dates where real stocks actually turned together.

Pipeline:
    1) Load every EGX .CA stock from the existing project DB.
    2) Detect fixed-rule major highs/lows for each stock.
    3) Group stocks by sector.
    4) Cluster nearby pivot dates into sector / whole-market time windows.
    5) Load the project's Egyptian market index/proxy and mark whether it also
       had a major pivot near the same window.
    6) Calculate GEO + HELIO planetary configurations only for those important
       windows, giving us a much smaller set of dates to compare with each
       stock's individual planetary DNA later.

Designed for the existing /app project:
    from core import database
    from monitor.gann_analysis import _daily_table, _load_egx_panel

Outputs include:
    all_stock_major_pivots.csv
    sector_turn_windows.csv
    market_turn_windows.csv
    IMPORTANT_MARKET_WINDOWS.csv
    important_windows_planetary_snapshot.csv

Run:
    python /app/egx_turn_window_map_v1.py \
      --output-dir /app/egx_turn_window_map_v1

Quick test:
    python /app/egx_turn_window_map_v1.py \
      --output-dir /app/egx_turn_window_test \
      --max-symbols 30
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import swisseph as swe
except Exception:
    swe = None


PLANETS = {
    "MERCURY": getattr(swe, "MERCURY", None) if swe else None,
    "VENUS": getattr(swe, "VENUS", None) if swe else None,
    "MARS": getattr(swe, "MARS", None) if swe else None,
    "JUPITER": getattr(swe, "JUPITER", None) if swe else None,
    "SATURN": getattr(swe, "SATURN", None) if swe else None,
    "URANUS": getattr(swe, "URANUS", None) if swe else None,
    "NEPTUNE": getattr(swe, "NEPTUNE", None) if swe else None,
    "PLUTO": getattr(swe, "PLUTO", None) if swe else None,
}


def pct(a,b):
    if not np.isfinite(a) or not np.isfinite(b) or a == 0:
        return np.nan
    return (b/a-1)*100.0


def angular_distance(a,b):
    d=abs((a-b)%360.0)
    return min(d,360.0-d)


def base_symbol(s):
    s=str(s).strip().upper()
    return s[:-3] if s.endswith(".CA") else s


def pick_col(t, aliases, required=True):
    m={c.name.lower():c for c in t.c}
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    if required:
        raise KeyError(f"Missing {aliases}; available={list(m)}")
    return None


def load_stocks(max_symbols=None):
    from sqlalchemy import select
    from core import database
    from monitor.gann_analysis import _daily_table

    DB=database()
    out={}
    with DB() as s:
        t=_daily_table(s)
        cs=pick_col(t,["symbol","ticker","sym"])
        cd=pick_col(t,["session_date","date","d","datetime","timestamp","ts"])
        cc=pick_col(t,["close","c","adj_close"])
        co=pick_col(t,["open","o"],False)
        ch=pick_col(t,["high","h"],False)
        cl=pick_col(t,["low","l"],False)

        syms=[r[0] for r in s.execute(
            select(cs).where(cs.ilike("%.CA")).distinct().order_by(cs)
        ).all() if r and r[0]]

        if max_symbols:
            syms=syms[:max_symbols]

        print("EGX symbols:",len(syms))

        for k,sym in enumerate(syms,1):
            cols=[cd,cc]
            names=["date","close"]
            for c,n in ((co,"open"),(ch,"high"),(cl,"low")):
                if c is not None:
                    cols.append(c); names.append(n)

            rows=s.execute(select(*cols).where(cs==sym).order_by(cd)).all()
            if not rows:
                continue

            df=pd.DataFrame(rows,columns=names)
            df["date"]=pd.to_datetime(df["date"],errors="coerce")
            for c in ["open","high","low","close"]:
                if c in df:
                    df[c]=pd.to_numeric(df[c],errors="coerce")
            if "open" not in df: df["open"]=df["close"]
            if "high" not in df: df["high"]=df["close"]
            if "low" not in df: df["low"]=df["close"]

            df=(df.dropna(subset=["date","open","high","low","close"])
                  .sort_values("date")
                  .drop_duplicates("date",keep="last")
                  .reset_index(drop=True))
            df=df[df["close"]>0].reset_index(drop=True)

            if len(df)>=250:
                out[str(sym).upper()]=df

            if k%25==0 or k==len(syms):
                print(f"  loaded {k}/{len(syms)}",flush=True)
    return out


def load_index():
    from core import database
    from monitor.gann_analysis import _load_egx_panel
    DB=database()
    idx,panel,source=_load_egx_panel(DB)
    x=idx[["d","c"]].copy()
    x.columns=["date","close"]
    x["date"]=pd.to_datetime(x["date"])
    x["open"]=x["close"]; x["high"]=x["close"]; x["low"]=x["close"]
    return x[["date","open","high","low","close"]].reset_index(drop=True),str(source)


def fetch_sector_map():
    try:
        import requests
        payload={
            "filter":[{"left":"exchange","operation":"equal","right":"EGX"}],
            "options":{"lang":"en"},
            "markets":["egypt"],
            "symbols":{"query":{"types":["stock"]},"tickers":[]},
            "columns":["name","description","sector","industry"],
            "range":[0,1000],
        }
        r=requests.post(
            "https://scanner.tradingview.com/egypt/scan",
            json=payload,timeout=30,headers={"User-Agent":"Mozilla/5.0"}
        )
        r.raise_for_status()
        ans={}
        for item in r.json().get("data",[]):
            d=item.get("d") or []
            if len(d)>=3:
                ans[base_symbol(d[0])]=str(d[2]).strip() if d[2] else "UNKNOWN"
        print("Sector labels:",len(ans))
        return ans
    except Exception as e:
        print("Sector map unavailable:",e)
        return {}


def detect_pivots(df,left=20,right=20,min_move=10.0,min_spacing=15):
    h=df["high"].to_numpy(float)
    l=df["low"].to_numpy(float)
    rows=[]

    for i in range(left,len(df)-right):
        if h[i]>=np.max(h[i-left:i]) and h[i]>np.max(h[i+1:i+1+right]):
            rows.append({"index":i,"date":df.loc[i,"date"],"type":"HIGH","price":h[i]})
        if l[i]<=np.min(l[i-left:i]) and l[i]<np.min(l[i+1:i+1+right]):
            rows.append({"index":i,"date":df.loc[i,"date"],"type":"LOW","price":l[i]})

    if not rows:
        return pd.DataFrame(columns=["index","date","type","price"])

    p=pd.DataFrame(rows).sort_values(["index","type"]).reset_index(drop=True)

    # same-type collapse
    z=[]
    for _,r in p.iterrows():
        d=r.to_dict()
        if not z or z[-1]["type"]!=d["type"]:
            z.append(d)
        else:
            old=z[-1]
            better=(d["type"]=="HIGH" and d["price"]>old["price"]) or \
                   (d["type"]=="LOW" and d["price"]<old["price"])
            if better: z[-1]=d

    # fixed move / spacing filter
    clean=[]
    for r in z:
        if not clean:
            clean.append(r); continue
        prev=clean[-1]
        m=abs(pct(float(prev["price"]),float(r["price"])))
        if int(r["index"])-int(prev["index"])<min_spacing:
            continue
        if not np.isfinite(m) or m<min_move:
            continue
        clean.append(r)

    return pd.DataFrame(clean)


def market_calendar(stocks,index_df):
    dates=set(pd.to_datetime(index_df["date"]).tolist())
    for df in stocks.values():
        dates.update(pd.to_datetime(df["date"]).tolist())
    return pd.DataFrame({"date":sorted(dates)}).reset_index(drop=True)


def add_cal_index(df,cal):
    if df.empty: return df.copy()
    m={pd.Timestamp(d):i for i,d in enumerate(cal["date"])}
    x=df.copy()
    x["cal_index"]=pd.to_datetime(x["date"]).map(m)
    return x.dropna(subset=["cal_index"]).assign(
        cal_index=lambda q:q["cal_index"].astype(int)
    )


def cluster_rows(df,gap=3):
    if df.empty: return []
    x=df.sort_values("cal_index").reset_index(drop=True)
    groups=[]; start=0
    for i in range(1,len(x)):
        if int(x.loc[i,"cal_index"])-int(x.loc[i-1,"cal_index"])>gap:
            groups.append(x.iloc[start:i].copy()); start=i
    groups.append(x.iloc[start:].copy())
    return groups


def eligible_count(asset_ranges,date,sector=None):
    x=asset_ranges.copy()
    if sector is not None:
        x=x[x["sector"]==sector]
    return int(((x["start_date"]<=date)&(x["end_date"]>=date)).sum())


def build_sector_windows(pivots,asset_ranges,cal,gap=3,min_symbols=3,min_breadth=.25):
    rows=[]
    for sec,sdf in pivots.groupby("sector"):
        for g in cluster_rows(sdf,gap):
            counts=g.groupby("cal_index")["symbol"].nunique().sort_values(ascending=False)
            ci=int(counts.index[0])
            d=pd.Timestamp(cal.iloc[ci]["date"])
            syms=sorted(set(g["symbol"]))
            elig=eligible_count(asset_ranges,d,sec)
            breadth=len(syms)/elig if elig else 0

            if len(syms)<min_symbols or breadth<min_breadth:
                continue

            highs=len(set(g.loc[g["type"]=="HIGH","symbol"]))
            lows=len(set(g.loc[g["type"]=="LOW","symbol"]))
            dom="HIGH" if highs>lows else ("LOW" if lows>highs else "MIXED")

            rows.append({
                "sector":sec,
                "date":d,
                "window_start":g["date"].min(),
                "window_end":g["date"].max(),
                "unique_stocks":len(syms),
                "eligible_sector_stocks":elig,
                "sector_breadth_pct":100*breadth,
                "high_stocks":highs,
                "low_stocks":lows,
                "dominant_turn":dom,
                "symbols":" ".join(syms),
            })
    return pd.DataFrame(rows)


def build_market_windows(pivots,asset_ranges,sector_windows,index_pivots,cal,
                         gap=3,min_stocks=8,min_breadth=.10):
    rows=[]
    idx_ci=index_pivots["cal_index"].to_numpy(int) if not index_pivots.empty else np.array([],dtype=int)

    for wid,g in enumerate(cluster_rows(pivots,gap),1):
        counts=g.groupby("cal_index")["symbol"].nunique().sort_values(ascending=False)
        ci=int(counts.index[0])
        d=pd.Timestamp(cal.iloc[ci]["date"])
        syms=sorted(set(g["symbol"]))
        elig=eligible_count(asset_ranges,d)
        breadth=len(syms)/elig if elig else 0

        if len(syms)<min_stocks or breadth<min_breadth:
            continue

        highs=len(set(g.loc[g["type"]=="HIGH","symbol"]))
        lows=len(set(g.loc[g["type"]=="LOW","symbol"]))
        dom="HIGH" if highs>lows else ("LOW" if lows>highs else "MIXED")

        if sector_windows.empty:
            near_sec=pd.DataFrame()
        else:
            sec_tmp=add_cal_index(sector_windows,cal)
            near_sec=sec_tmp[np.abs(sec_tmp["cal_index"]-ci)<=gap]

        strong_secs=sorted(set(near_sec["sector"])) if not near_sec.empty else []
        idx_hit=bool(len(idx_ci) and np.min(np.abs(idx_ci-ci))<=gap)

        if idx_hit and len(strong_secs)>=2:
            category="MARKET+INDEX+MULTI_SECTOR"
        elif idx_hit:
            category="MARKET+INDEX"
        elif len(strong_secs)>=2:
            category="MARKET+MULTI_SECTOR"
        else:
            category="MARKET"

        score=(
            100*breadth
            + 8*min(len(strong_secs),5)
            + (20 if idx_hit else 0)
            + 0.5*len(syms)
        )

        rows.append({
            "window_id":wid,
            "date":d,
            "window_start":g["date"].min(),
            "window_end":g["date"].max(),
            "category":category,
            "confluence_score":score,
            "unique_stocks":len(syms),
            "eligible_market_stocks":elig,
            "market_breadth_pct":100*breadth,
            "high_stocks":highs,
            "low_stocks":lows,
            "dominant_turn":dom,
            "strong_sector_count":len(strong_secs),
            "strong_sectors":" | ".join(strong_secs),
            "index_pivot_confirmed":int(idx_hit),
            "symbols":" ".join(syms),
        })
    return pd.DataFrame(rows)


def planetary_snapshot(dates):
    if swe is None:
        return pd.DataFrame()

    rows=[]
    for d in sorted(set(pd.to_datetime(dates))):
        j=swe.julday(d.year,d.month,d.day,12.0)
        row={"date":d}

        positions={}
        for system,helio in (("GEO",False),("HELIO",True)):
            positions[system]={}
            flags=swe.FLG_SWIEPH|swe.FLG_SPEED
            if helio: flags|=swe.FLG_HELCTR

            for name,pid in PLANETS.items():
                xx,_=swe.calc_ut(j,pid,flags)
                lon=float(xx[0])%360.0
                spd=float(xx[3])
                positions[system][name]=lon
                row[f"{system}_{name}_lon"]=lon
                row[f"{system}_{name}_retro"]=int(spd<0)

            # store all pair angles; later we can identify repeating fingerprints
            names=list(PLANETS)
            for i in range(len(names)):
                for k in range(i+1,len(names)):
                    a,b=names[i],names[k]
                    row[f"{system}_{a}_{b}_angle"]=angular_distance(
                        positions[system][a],positions[system][b]
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    print("[1/7] Loading stocks + market index...")
    stocks=load_stocks(args.max_symbols)
    index_df,index_source=load_index()
    sectors=fetch_sector_map()

    print("[2/7] Detecting major stock pivots...")
    pivot_frames=[]
    ranges=[]

    for k,(sym,df) in enumerate(stocks.items(),1):
        p=detect_pivots(
            df,args.pivot_left,args.pivot_right,
            args.min_pivot_move,args.min_pivot_spacing
        )
        sec=sectors.get(base_symbol(sym),"UNKNOWN")
        if not p.empty:
            p["symbol"]=sym
            p["sector"]=sec
            pivot_frames.append(p)

        ranges.append({
            "symbol":sym,"sector":sec,
            "start_date":df["date"].min(),
            "end_date":df["date"].max(),
            "rows":len(df),
            "major_pivots":len(p),
        })
        if k%25==0 or k==len(stocks):
            print(f"  pivots {k}/{len(stocks)}",flush=True)

    pivots=pd.concat(pivot_frames,ignore_index=True) if pivot_frames else pd.DataFrame()
    ranges=pd.DataFrame(ranges)

    print("[3/7] Detecting index/proxy pivots...")
    idxp=detect_pivots(
        index_df,args.pivot_left,args.pivot_right,
        max(3.0,args.min_pivot_move/2),args.min_pivot_spacing
    )
    idxp["symbol"]="EGX_MARKET_INDEX"
    idxp["sector"]="INDEX"

    print("[4/7] Building common market calendar...")
    cal=market_calendar(stocks,index_df)
    pivots=add_cal_index(pivots,cal)
    idxp=add_cal_index(idxp,cal)

    print("[5/7] Finding sector + market turn windows...")
    sector_windows=build_sector_windows(
        pivots,ranges,cal,args.window_gap,
        args.min_sector_symbols,args.min_sector_breadth
    )
    market_windows=build_market_windows(
        pivots,ranges,sector_windows,idxp,cal,args.window_gap,
        args.min_market_symbols,args.min_market_breadth
    )

    if market_windows.empty:
        important=pd.DataFrame()
    else:
        important=market_windows[
            market_windows["category"].isin([
                "MARKET+INDEX+MULTI_SECTOR",
                "MARKET+INDEX",
                "MARKET+MULTI_SECTOR",
            ])
        ].copy()
        if important.empty:
            important=market_windows.nlargest(min(50,len(market_windows)),"confluence_score")
        important=important.sort_values("date").reset_index(drop=True)

    print("[6/7] Calculating planetary snapshots only on important dates...")
    planet=planetary_snapshot(important["date"] if not important.empty else [])

    print("[7/7] Saving...")
    ranges.to_csv(outdir/"asset_ranges_and_pivot_counts.csv",index=False)
    pivots.to_csv(outdir/"all_stock_major_pivots.csv",index=False)
    idxp.to_csv(outdir/"index_major_pivots.csv",index=False)
    sector_windows.to_csv(outdir/"sector_turn_windows.csv",index=False)
    market_windows.to_csv(outdir/"market_turn_windows.csv",index=False)
    important.to_csv(outdir/"IMPORTANT_MARKET_WINDOWS.csv",index=False)
    planet.to_csv(outdir/"important_windows_planetary_snapshot.csv",index=False)
    cal.to_csv(outdir/"market_calendar.csv",index=False)

    report={
        "index_source":index_source,
        "stocks_loaded":len(stocks),
        "stock_major_pivots":len(pivots),
        "index_major_pivots":len(idxp),
        "sector_turn_windows":len(sector_windows),
        "market_turn_windows":len(market_windows),
        "important_market_windows":len(important),
        "planetary_snapshots":len(planet),
    }
    (outdir/"run_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\n=== EGX TURN WINDOW MAP V1 ===")
    for k,v in report.items():
        print(f"{k}: {v}")

    print("\n=== TOP MARKET WINDOWS ===")
    if market_windows.empty:
        print("NONE")
    else:
        cols=[
            "date","category","confluence_score","unique_stocks",
            "eligible_market_stocks","market_breadth_pct",
            "dominant_turn","strong_sector_count",
            "index_pivot_confirmed","strong_sectors"
        ]
        print(
            market_windows.sort_values("confluence_score",ascending=False)
            [cols].head(40).to_string(index=False)
        )

    print("\nMost important:")
    print(outdir/"IMPORTANT_MARKET_WINDOWS.csv")
    print(outdir/"important_windows_planetary_snapshot.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="/app/egx_turn_window_map_v1")
    p.add_argument("--max-symbols",type=int,default=None)
    p.add_argument("--pivot-left",type=int,default=20)
    p.add_argument("--pivot-right",type=int,default=20)
    p.add_argument("--min-pivot-move",type=float,default=10.0)
    p.add_argument("--min-pivot-spacing",type=int,default=15)
    p.add_argument("--window-gap",type=int,default=3)
    p.add_argument("--min-sector-symbols",type=int,default=3)
    p.add_argument("--min-sector-breadth",type=float,default=0.25)
    p.add_argument("--min-market-symbols",type=int,default=8)
    p.add_argument("--min-market-breadth",type=float,default=0.10)
    main(p.parse_args())
