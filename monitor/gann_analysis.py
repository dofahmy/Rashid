
from __future__ import annotations
import math
import time
import bisect
from datetime import date, datetime, timedelta
from typing import Any
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select, func, and_

IMPORTANT_RATIOS=[(0.25,"1/4",82),(1/3,"1/3",78),(0.375,"3/8",72),(0.50,"1/2",92),(0.625,"5/8",72),(2/3,"2/3",78),(0.75,"3/4",82),(0.875,"7/8",72),(1.00,"1/1",95)]
SQ9_ANGLES=[(45,78),(90,90),(135,78),(180,94),(225,78),(270,90),(315,78),(360,96)]
MASTER_144=[(36,86),(45,80),(48,78),(54,80),(63,82),(72,94),(81,82),(90,90),(96,78),(108,86),(117,76),(126,78),(135,82),(144,98)]
MIKULA_225_CELL_COUNTS=[9,25,49,81,121,169,225,289,361]

# Egyptian market anchor settings.
EGX_INDEX_SYMBOL="^CASE30"
EGX_MARKET_WING=10               # index pivot confirmation / local extremum wing
EGX_MARKET_EVENT_WINDOW=3        # stocks may bottom/top within ±3 sessions of index
EGX_MARKET_LOCAL_WINDOW=12       # local stock extremum reference window
EGX_MARKET_MIN_SCORE=90.0        # accept only market-wide pivots
EGX_MARKET_MIN_SWING_PCT=15.0     # minimum index swing between accepted opposite pivots
EGX_MARKET_MIN_SEPARATION=25     # sessions between same-type major pivots
EGX_LEADER_COUNT=15
EGX_MARKET_FOLLOW_WINDOW=40       # sessions after candidate used to confirm follow-through
EGX_MARKET_FOLLOW_MIN_PCT=10.0   # required move away from the pivot
EGX_MARKET_LOW_MIN_BREADTH=55.0  # principal market lows should be broad
EGX_MARKET_HIGH_MIN_BREADTH=30.0 # highs can be narrower than panic/capitulation lows
EGX_MARKET_HISTORY_DATES=2500     # roughly full 2019+ history
EGX_MARKET_MIN_EVENT_GAP=20        # minimum sessions between accepted LOW/HIGH pivots
EGX_MARKET_CLOSE_WING=10           # close-based pivot wing on synthetic/official index
EGX_FINAL_CONSENSUS_MIN_VOTES=3     # selected by at least 3 of 5 reasonable models
EGX_FINAL_MIN_GAP=15               # absolute safety floor between opposite pivots
EGX_FINAL_MIN_TRANSITION_SWING=6.0 # only removes market noise, not a 'major' hard threshold

GANN_ENGINE_VERSION="5.0 FINAL"
FINAL_TIME_HORIZON_DAYS=270
FINAL_TIME_WINDOW_DAYS=2
FINAL_TIME_MIN_SEPARATION_DAYS=7
FINAL_LEVEL_MIN_SEPARATION_PCT=1.75
FINAL_BACKTEST_MIN_SAMPLES=3
FINAL_HISTORY_LIMIT_EGX=2200
FINAL_HISTORY_LIMIT_US=1400

# Major/representative listed banks. Only symbols available in the database count.
EGX_BANK_SYMBOLS={
    "COMI.CA","QNBA.CA","ADIB.CA","CIEB.CA","FAIT.CA",
    "HDBK.CA","EXPA.CA","CANA.CA","SAUD.CA","EGBE.CA",
}

_EGX_MARKET_CACHE={"stamp":0.0,"key":None,"pivots":[]}


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


def _build_synthetic_egx30(panel,leader_count=30):
    """Build a robust market-index proxy from the 30 most liquid Egyptian shares.

    Used only when the external ^CASE30 history feed is unavailable.
    Each constituent is rebased to 100 from its first close in the loaded
    window, then the normalized OHLC series are equal-weighted by date.
    This is NOT the official EGX30 calculation; it is a market-turn proxy.
    """
    if panel.empty:
        return pd.DataFrame(),[]

    p=panel.copy().sort_values(["symbol","d"]).reset_index(drop=True)
    p["dv"]=pd.to_numeric(p["c"],errors="coerce")*pd.to_numeric(p["v"],errors="coerce").fillna(0)

    # Prefer persistent liquidity, not one-day spikes.
    med=p.groupby("symbol")["dv"].median().sort_values(ascending=False)
    leaders=list(med.head(leader_count).index)
    q=p[p["symbol"].isin(leaders)].copy()
    if q.empty:
        return pd.DataFrame(),[]

    first=q.groupby("symbol")["c"].transform("first")
    first=first.replace(0,np.nan)
    for col in ["o","h","l","c"]:
        q[f"n_{col}"]=100.0*pd.to_numeric(q[col],errors="coerce")/first

    agg=q.groupby("d").agg(
        o=("n_o","mean"),
        h=("n_h","mean"),
        l=("n_l","mean"),
        c=("n_c","mean"),
        v=("v","sum"),
        members=("symbol","nunique"),
    ).reset_index()

    # Require enough members so sparse early dates do not create fake pivots.
    agg=agg[agg["members"]>=max(10,int(leader_count*0.50))].copy()
    agg=agg.sort_values("d").reset_index(drop=True)
    return agg[["d","o","h","l","c","v"]],leaders


def _load_egx_panel(DB,limit_dates=EGX_MARKET_HISTORY_DATES):
    """Load official EGX30 if available; otherwise synthesize a market proxy.

    Returns (index_df, stock_panel, source_label).
    """
    with DB() as s:
        t=_daily_table(s)

        # Load Egyptian stocks independently of index availability.
        recent_dates=list(s.scalars(
            select(t.c.session_date)
            .where(t.c.symbol.ilike("%.CA"))
            .distinct()
            .order_by(t.c.session_date.desc())
            .limit(limit_dates)
        ).all())
        if not recent_dates:
            return pd.DataFrame(),pd.DataFrame(),"NONE"

        min_date=min(str(x) for x in recent_dates)
        stock_rows=s.execute(
            select(t.c.symbol,t.c.session_date,t.c.o,t.c.h,t.c.l,t.c.c,t.c.v)
            .where(t.c.symbol.ilike("%.CA"),t.c.session_date>=min_date)
            .order_by(t.c.session_date,t.c.symbol)
        ).all()

        idx_rows=s.execute(
            select(t.c.session_date,t.c.o,t.c.h,t.c.l,t.c.c,t.c.v)
            .where(t.c.symbol==EGX_INDEX_SYMBOL,t.c.session_date>=min_date)
            .order_by(t.c.session_date)
        ).all()

    panel=pd.DataFrame(stock_rows,columns=["symbol","d","o","h","l","c","v"])
    if panel.empty:
        return pd.DataFrame(),pd.DataFrame(),"NONE"
    panel["d"]=pd.to_datetime(panel["d"]).dt.date
    for c in ["o","h","l","c","v"]:
        panel[c]=pd.to_numeric(panel[c],errors="coerce")
    panel=panel.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
    panel["dv"]=panel["c"]*panel["v"].fillna(0)

    if idx_rows:
        idx=pd.DataFrame(idx_rows,columns=["d","o","h","l","c","v"])
        idx["d"]=pd.to_datetime(idx["d"]).dt.date
        for c in ["o","h","l","c","v"]:
            idx[c]=pd.to_numeric(idx[c],errors="coerce")
        idx=idx.dropna(subset=["h","l","c"]).reset_index(drop=True)
        return idx,panel,"OFFICIAL_CASE30"

    # Railway/Yahoo chart history for ^CASE30 can return no rows even when
    # individual .CA shares work. Fall back to a market-wide Top-30 liquid proxy.
    idx,leaders=_build_synthetic_egx30(panel,leader_count=30)
    return idx,panel,"SYNTHETIC_TOP30_LIQUID"


def _index_candidate_pivots(idx,wing=EGX_MARKET_CLOSE_WING):
    """Candidate turns from the market close series, not synthetic high/low.

    For a synthetic breadth index, averaging highs/lows across different stocks
    can create artificial one-day ranges. Close is a coherent single series and
    is therefore used to locate the market turning date. Breadth/leaders/banks
    confirm the turn afterwards.
    """
    if len(idx)<wing*2+10:
        return []
    c=idx["c"].to_numpy(float)
    out=[]
    for i in range(wing,len(idx)-wing):
        window=c[i-wing:i+wing+1]
        if not np.isfinite(c[i]):
            continue
        if c[i] <= np.nanmin(window):
            out.append({"i":i,"kind":"L","date":idx.iloc[i]["d"],"price":float(c[i])})
        if c[i] >= np.nanmax(window):
            out.append({"i":i,"kind":"H","date":idx.iloc[i]["d"],"price":float(c[i])})
    return sorted(out,key=lambda x:(x["i"],0 if x["kind"]=="L" else 1))


def _prepare_symbol_turn_data(panel):
    """Prepare each EGX symbol once for very fast pivot confirmation.

    The previous implementation filtered the full ~385k-row panel separately
    for every symbol and every candidate pivot. That is correct but extremely
    slow. Here we group once and keep compact Python/numpy arrays.
    """
    out={}
    if panel.empty:
        return out
    for sym,sdf in panel.groupby("symbol",sort=False):
        sdf=sdf.sort_values("d")
        dates=list(sdf["d"])
        if len(dates)<20:
            continue
        out[str(sym)]={
            "dates":dates,
            "ord":[d.toordinal() for d in dates],
            "h":sdf["h"].to_numpy(float),
            "l":sdf["l"].to_numpy(float),
        }
    return out

def _nearest_pos(ordinals,target_ord):
    """Nearest sorted ordinal position using binary search."""
    n=len(ordinals)
    if n==0:return None
    j=bisect.bisect_left(ordinals,target_ord)
    if j<=0:return 0
    if j>=n:return n-1
    return j-1 if abs(ordinals[j-1]-target_ord)<=abs(ordinals[j]-target_ord) else j

def _symbol_confirms_market_turn_fast(data,event_date,kind,
                                      event_window=EGX_MARKET_EVENT_WINDOW,
                                      local_window=EGX_MARKET_LOCAL_WINDOW):
    if not data:
        return False
    j=_nearest_pos(data["ord"],event_date.toordinal())
    if j is None:
        return False
    n=len(data["dates"])
    lo=max(0,j-local_window); hi=min(n,j+local_window+1)
    elo=max(0,j-event_window); ehi=min(n,j+event_window+1)
    if hi<=lo or ehi<=elo:
        return False

    if kind=="L":
        local_ext=float(np.nanmin(data["l"][lo:hi]))
        event_ext=float(np.nanmin(data["l"][elo:ehi]))
        return math.isfinite(local_ext) and math.isfinite(event_ext) and event_ext<=local_ext*1.01

    local_ext=float(np.nanmax(data["h"][lo:hi]))
    event_ext=float(np.nanmax(data["h"][elo:ehi]))
    return math.isfinite(local_ext) and math.isfinite(event_ext) and event_ext>=local_ext*0.99

def _leaders_for_date(panel,event_date,count=EGX_LEADER_COUNT):
    """Dynamic leaders = highest median EGP turnover in prior ~120 calendar days."""
    if panel.empty:return []
    start=event_date-timedelta(days=120)
    sub=panel[(panel["d"]<event_date)&(panel["d"]>=start)]
    if sub.empty:return []
    med=sub.groupby("symbol",sort=False)["dv"].median().nlargest(count)
    return list(med.index)

def _confirmation_rate_prepared(prepared,symbols,event_date,kind):
    """Cross-sectional confirmation rate without repeatedly filtering DataFrames."""
    if not prepared or not symbols:
        return 0.0,0,0
    good=0;avail=0
    for sym in symbols:
        data=prepared.get(str(sym))
        if not data:
            continue
        avail+=1
        if _symbol_confirms_market_turn_fast(data,event_date,kind):
            good+=1
    return (good/avail if avail else 0.0),good,avail


def _sat(value,target):
    """Smooth 0..100 saturation score; no cliff at the target."""
    try:
        v=max(0.0,float(value))
        t=max(1e-9,float(target))
    except Exception:
        return 0.0
    return min(100.0,100.0*v/t)

def _candidate_quality(row,profile="balanced"):
    """Market-wide significance score under several reasonable viewpoints.

    LOWs emphasize breadth/capitulation.
    HIGHs emphasize index reversal + leaders + banks, matching the intended
    definition that important tops are visible in the index and large/bank stocks.
    """
    kind=row["kind"]
    prior=_sat(row.get("prior_swing_pct",0),15)
    follow=_sat(row.get("follow_pct",0),12)
    breadth=_sat(row.get("breadth_pct",0),60 if kind=="L" else 42)
    leaders=_sat(row.get("leaders_pct",0),55)
    banks=_sat(row.get("banks_pct",0),55)

    profiles={
        "balanced":{
            "L":(20,20,35,15,10),
            "H":(20,20,20,20,20),
        },
        "breadth_led":{
            "L":(15,15,45,15,10),
            "H":(15,15,30,20,20),
        },
        "leadership_led":{
            "L":(15,15,25,20,25),
            "H":(15,15,15,25,30),
        },
        "reversal_led":{
            "L":(25,30,25,10,10),
            "H":(25,30,15,15,15),
        },
        "conservative":{
            "L":(20,25,30,15,10),
            "H":(25,25,15,17.5,17.5),
        },
    }
    wp,wf,wb,wl,wbank=profiles[profile][kind]
    q=(wp*prior+wf*follow+wb*breadth+wl*leaders+wbank*banks)/100.0
    return round(q,2)

def _build_raw_market_candidates(DB):
    """Build close-pivot candidates and all confirmation metrics once."""
    idx,panel,index_source=_load_egx_panel(DB)
    if idx.empty or panel.empty:
        return idx,panel,index_source,[]

    candidates=_index_candidate_pivots(idx)
    all_symbols=sorted(panel["symbol"].dropna().unique().tolist())
    prepared=_prepare_symbol_turn_data(panel)
    all_symbols=[s for s in all_symbols if s in prepared]
    rows=[]

    for c in candidates:
        i=c["i"];event_date=c["date"];kind=c["kind"]

        back=idx.iloc[max(0,i-90):i+1]
        if kind=="L":
            opp=float(back["c"].max()) if not back.empty else c["price"]
            prior_swing=max(0.0,(opp/c["price"]-1)*100) if c["price"] else 0.0
        else:
            opp=float(back["c"].min()) if not back.empty else c["price"]
            prior_swing=max(0.0,(c["price"]/opp-1)*100) if opp else 0.0

        fwd=idx.iloc[i+1:min(len(idx),i+1+EGX_MARKET_FOLLOW_WINDOW)]
        if fwd.empty:
            continue
        if kind=="L":
            fwd_ext=float(fwd["c"].max())
            follow_pct=max(0.0,(fwd_ext/c["price"]-1)*100) if c["price"] else 0.0
        else:
            fwd_ext=float(fwd["c"].min())
            follow_pct=max(0.0,(1-fwd_ext/c["price"])*100) if c["price"] else 0.0

        breadth_rate,breadth_good,breadth_n=_confirmation_rate_prepared(
            prepared,all_symbols,event_date,kind
        )
        leaders=_leaders_for_date(panel,event_date)
        leader_rate,leader_good,leader_n=_confirmation_rate_prepared(
            prepared,leaders,event_date,kind
        )
        banks=sorted(EGX_BANK_SYMBOLS.intersection(all_symbols))
        bank_rate,bank_good,bank_n=_confirmation_rate_prepared(
            prepared,banks,event_date,kind
        )

        available_i=min(len(idx)-1,i+EGX_MARKET_WING)
        row={
            **c,
            "prior_swing_pct":round(prior_swing,2),
            "follow_pct":round(follow_pct,2),
            "breadth_pct":round(100*breadth_rate,1),
            "breadth_count":breadth_good,"breadth_total":breadth_n,
            "leaders_pct":round(100*leader_rate,1),
            "leaders_count":leader_good,"leaders_total":leader_n,
            "banks_pct":round(100*bank_rate,1),
            "banks_count":bank_good,"banks_total":bank_n,
            "available_date":idx.iloc[available_i]["d"],
            "index_source":index_source,
        }
        for prof in ("balanced","breadth_led","leadership_led","reversal_led","conservative"):
            row[f"quality_{prof}"]=_candidate_quality(row,prof)
        row["quality"]=row["quality_balanced"]
        rows.append(row)

    return idx,panel,index_source,rows

def _transition_swing(a,b):
    if not a or not b or not a.get("price"):
        return 0.0
    return abs(float(b["price"])/float(a["price"])-1)*100.0

def _select_profile_sequence(candidates,quality_key,min_gap,min_swing,node_cost):
    """Dynamic-programming selection of an alternating market-cycle sequence.

    Unlike greedy filters, a later candidate cannot erase a valid intermediate
    turn simply because another threshold failed. The optimizer rewards:
    - stable market-wide quality,
    - meaningful opposite-direction swing,
    - reasonable time separation,
    while charging a cost for every extra pivot to avoid over-segmentation.
    """
    cs=sorted(candidates,key=lambda x:(x["i"],x["kind"]))
    n=len(cs)
    if not n:
        return []

    dp=[-1e18]*n
    prev=[None]*n

    for j,b in enumerate(cs):
        q=float(b.get(quality_key,0))
        base=q-node_cost
        dp[j]=base

        for i in range(j):
            a=cs[i]
            if a["kind"]==b["kind"]:
                continue
            gap=int(b["i"])-int(a["i"])
            if gap<min_gap:
                continue
            swing=_transition_swing(a,b)
            if swing<min_swing:
                continue

            # Soft bonuses: large swings/time separation help, but neither
            # creates a cliff just above an arbitrary threshold.
            swing_bonus=min(24.0,1.15*swing)
            gap_bonus=min(6.0,max(0.0,(gap-min_gap)/25.0))
            val=dp[i]+base+swing_bonus+gap_bonus
            if val>dp[j]:
                dp[j]=val
                prev[j]=i

    j=max(range(n),key=lambda k:dp[k])
    seq=[]
    while j is not None:
        seq.append(cs[j])
        j=prev[j]
    seq.reverse()

    # Drop negative-value singleton/noisy starts until sequence is meaningful.
    # Retain at least two pivots if possible.
    while len(seq)>2 and float(seq[0].get(quality_key,0))<55:
        seq=seq[1:]
    return seq

def _consensus_major_pivots(DB,include_diagnostics=False):
    idx,panel,index_source,raw=_build_raw_market_candidates(DB)
    if not raw:
        return ([],[]) if include_diagnostics else []

    # Five deliberately different but reasonable decision models.
    configs=[
        ("balanced","quality_balanced",15,7.0,62.0),
        ("breadth_led","quality_breadth_led",15,6.0,60.0),
        ("leadership_led","quality_leadership_led",18,6.0,62.0),
        ("reversal_led","quality_reversal_led",18,8.0,64.0),
        ("conservative","quality_conservative",22,10.0,68.0),
    ]

    votes={(r["date"],r["kind"]):0 for r in raw}
    profile_hits={(r["date"],r["kind"]):[] for r in raw}

    for name,qkey,gap,swing,cost in configs:
        seq=_select_profile_sequence(raw,qkey,gap,swing,cost)
        for x in seq:
            k=(x["date"],x["kind"])
            votes[k]=votes.get(k,0)+1
            profile_hits.setdefault(k,[]).append(name)

    ranked=[]
    for r in raw:
        x=dict(r)
        k=(x["date"],x["kind"])
        x["consensus_votes"]=int(votes.get(k,0))
        x["stability_pct"]=round(100*x["consensus_votes"]/len(configs),1)
        x["selected_profiles"]=profile_hits.get(k,[])
        # Blend balanced significance with stability. Stability is deliberately
        # large: a date that survives multiple reasonable models is preferable
        # to one created by a single parameter choice.
        x["final_score"]=round(
            0.60*float(x["quality_balanced"])+0.40*x["stability_pct"],1
        )
        ranked.append(x)

    stable=[x for x in ranked if x["consensus_votes"]>=EGX_FINAL_CONSENSUS_MIN_VOTES]

    # Final sequence uses stable candidates only and very mild safety floors.
    # No 15%/10% hard cliff remains here.
    stable.sort(key=lambda x:(x["i"],x["kind"]))
    final=[]
    for p in stable:
        if not final:
            final.append(p)
            continue

        prev=final[-1]
        if p["kind"]==prev["kind"]:
            # Same market phase: retain the more stable/significant extreme.
            more_extreme=(p["kind"]=="L" and p["price"]<prev["price"]) or \
                         (p["kind"]=="H" and p["price"]>prev["price"])
            if (p["final_score"]>prev["final_score"]+4 or
                (more_extreme and p["final_score"]>=prev["final_score"]-2)):
                final[-1]=p
            continue

        gap=int(p["i"])-int(prev["i"])
        swing=_transition_swing(prev,p)
        if gap<EGX_FINAL_MIN_GAP or swing<EGX_FINAL_MIN_TRANSITION_SWING:
            # Do not silently lose it from diagnostics; it remains ranked but
            # is not a principal cycle boundary.
            continue
        q=dict(p)
        q["swing_from_prev_pct"]=round(swing,2)
        final.append(q)

    # If an unstable but exceptional candidate lies between two final pivots,
    # allow rescue only when cross-sectional evidence is extraordinary.
    # This prevents long gaps such as 2024->2026 without reopening the door to noise.
    rescued=[]
    if len(final)>=2:
        for a,b in zip(final[:-1],final[1:]):
            rescued.append(a)
            between=[
                x for x in ranked
                if a["i"]<x["i"]<b["i"] and x["kind"]!=a["kind"]
                and x["kind"]!=b["kind"] if False
            ]
            # Because accepted endpoints alternate, a genuine missing pivot
            # would create same-type endpoints only; handled below.
        rescued.append(final[-1])

    # Repair same-type long gaps using one extraordinary market-wide candidate.
    # Usually final is already alternating, but this protects against consensus
    # dropping the only opposite turn in a long market cycle.
    repaired=[]
    for p in final:
        if not repaired:
            repaired.append(p);continue
        if p["kind"]!=repaired[-1]["kind"]:
            repaired.append(p);continue
        a=repaired[-1]
        between=[
            x for x in ranked
            if a["i"]<x["i"]<p["i"] and x["kind"]!=a["kind"]
            and (
                x["breadth_pct"]>=65 or
                (x["leaders_pct"]>=60 and x["banks_pct"]>=55) or
                x["quality_balanced"]>=88
            )
        ]
        if between:
            mid=max(between,key=lambda x:(x["final_score"],x["quality_balanced"]))
            repaired.append(mid)
        repaired.append(p)

    final=repaired[-14:]

    # Attach explicit decision reasons to every candidate.
    final_keys={(x["date"],x["kind"]) for x in final}
    diagnostics=[]
    for x in ranked:
        y=dict(x)
        if (x["date"],x["kind"]) in final_keys:
            y["decision"]="ACCEPTED"
            y["decision_reason"]=f'consensus {x["consensus_votes"]}/5; final_score={x["final_score"]}'
        elif x["consensus_votes"]<EGX_FINAL_CONSENSUS_MIN_VOTES:
            y["decision"]="REJECTED"
            y["decision_reason"]=f'consensus only {x["consensus_votes"]}/5'
        else:
            y["decision"]="REJECTED"
            y["decision_reason"]="stable candidate but lost to stronger same-cycle pivot / minimum cycle separation"
        diagnostics.append(y)

    if include_diagnostics:
        return final,diagnostics
    return final

def _major_market_pivots_uncached(DB,include_diagnostics=False):
    # Backward-compatible public name used by diagnostics and the Gann engine.
    return _consensus_major_pivots(DB,include_diagnostics=include_diagnostics)

def get_egx_market_pivots(DB,cache_seconds=1800):
    """Cached final consensus EGX major pivots."""
    now_ts=time.time()
    key=id(DB)
    if (_EGX_MARKET_CACHE.get("key")==key
        and now_ts-_EGX_MARKET_CACHE.get("stamp",0)<cache_seconds):
        return _EGX_MARKET_CACHE.get("pivots",[])
    piv=_major_market_pivots_uncached(DB)
    _EGX_MARKET_CACHE.update({"stamp":now_ts,"key":key,"pivots":piv})
    return piv

def _map_market_pivots_to_stock(df,market_pivots,event_window=EGX_MARKET_EVENT_WINDOW):
    """Use each major market date, then anchor to this stock's local extreme near it."""
    if df.empty:return []
    out=[]
    dates=list(df["d"])
    if not dates:return []
    first_d,last_d=dates[0],dates[-1]
    for mp in market_pivots:
        d=mp["date"]
        if d < first_d-timedelta(days=7) or d > last_d+timedelta(days=7):
            continue
        j=min(range(len(dates)),key=lambda k:abs((dates[k]-d).days))
        lo=max(0,j-event_window);hi=min(len(df)-1,j+event_window)
        w=df.iloc[lo:hi+1]
        if w.empty:continue
        if mp["kind"]=="L":
            rel=int(np.nanargmin(w["l"].to_numpy(float)))
            price=float(w.iloc[rel]["l"])
        else:
            rel=int(np.nanargmax(w["h"].to_numpy(float)))
            price=float(w.iloc[rel]["h"])
        ii=lo+rel
        out.append({
            "i":ii,
            "kind":mp["kind"],
            "price":price,
            "date":df.iloc[ii]["d"],
            "market_date":mp["date"],
            "available_date":mp["available_date"],
            "market_score":mp.get("final_score",mp.get("quality_balanced",mp.get("score",0))),
            "breadth_pct":mp["breadth_pct"],
            "leaders_pct":mp["leaders_pct"],
            "banks_pct":mp["banks_pct"],
            "index_price":mp["price"],
            "index_swing_pct":mp.get("prior_swing_pct",0),
            "consensus_votes":mp.get("consensus_votes",0),
            "stability_pct":mp.get("stability_pct",0),
            "final_score":mp.get("final_score",mp.get("quality",0)),
        })
    return out

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
        anchor_kinds=sorted(set(str(i.get("pivot_kind","")) for i in c["items"] if i.get("pivot_kind")))
        out.append({"level":round(c["level"],4),"strength":st,"methods":methods,"labels":[i["submethod"] for i in c["items"]],"side":"resistance" if c["level"]>current else "support","distance_pct":round(100*(c["level"]/current-1),2),"count":len(c["items"]),"anchor_kinds":anchor_kinds})
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
        anchor_kinds=sorted(set(str(i.get("anchor_kind","")) for i in c["items"] if i.get("anchor_kind")))
        out.append({"date":c["date"],"strength":st,"methods":methods,"labels":[i["submethod"] for i in c["items"]],"count":len(c["items"]),"anchor_kinds":anchor_kinds})
    return out

def strength_color(v):
    v=int(v or 0)
    return "#b91c1c" if v>=90 else "#ea580c" if v>=80 else "#ca8a04" if v>=70 else "#2563eb" if v>=60 else "#64748b"

def _main_level_score(x):
    """Geometry score for a clustered price level (not probability)."""
    strength=float(x.get("strength") or 0)
    count=max(1,int(x.get("count") or 1))
    methods=len(x.get("methods") or [])
    dist=abs(float(x.get("distance_pct") or 0))
    diversity=min(12,3.0*methods+1.5*(count-1))
    proximity_penalty=min(22,dist*0.32)
    return strength+diversity-proximity_penalty

def select_main_price_levels(pc,current,side,limit=2,atr=0.0):
    """Sparse decision levels: strong, reasonably near, and materially separated."""
    xs=[x for x in pc if (x["level"]>current if side=="resistance" else x["level"]<current)]
    if not xs:return []
    near=[x for x in xs if abs(float(x.get("distance_pct") or 0))<=45]
    if len(near)>=limit:xs=near
    ranked=sorted(xs,key=lambda x:(-_main_level_score(x),abs(float(x.get("distance_pct") or 0))))
    atr_pct=(100*atr/current) if current and atr else 0.0
    min_sep=max(FINAL_LEVEL_MIN_SEPARATION_PCT,1.10*atr_pct)
    picked=[]
    for x in ranked:
        if any(abs(x["level"]/q["level"]-1)*100<min_sep for q in picked if q.get("level")):
            continue
        y=dict(x)
        y["geometry_score"]=int(round(max(0,min(100,_main_level_score(x)))))
        picked.append(y)
        if len(picked)>=limit:break
    picked.sort(key=lambda x:x["level"],reverse=(side!="resistance"))
    return picked

def _tag_times(items,anchor_kind):
    out=[]
    for x in items:
        y=dict(x);y["anchor_kind"]=anchor_kind;out.append(y)
    return out

def select_final_time_windows(tc,kind=None,limit=2,horizon_days=FINAL_TIME_HORIZON_DAYS):
    """Strong near-term Gann time clusters, separated enough to be actionable."""
    today=date.today();end=today+timedelta(days=horizon_days)
    xs=[x for x in tc if today<=x["date"]<=end]
    preferred="L" if kind=="TOP" else "H" if kind=="LOW" else None
    def compat(x):
        ks=set(x.get("anchor_kinds") or [])
        if not preferred:return 0
        if preferred in ks:return 2
        if "RANGE" in ks:return 1
        return 0
    ranked=sorted(xs,key=lambda x:(-compat(x),-float(x.get("strength",0)),x["date"]))
    picked=[]
    for x in ranked:
        if any(abs((x["date"]-q["date"]).days)<FINAL_TIME_MIN_SEPARATION_DAYS for q in picked):
            continue
        y=dict(x)
        y["window_start"]=x["date"]-timedelta(days=FINAL_TIME_WINDOW_DAYS)
        y["window_end"]=x["date"]+timedelta(days=FINAL_TIME_WINDOW_DAYS)
        y["direction_compatibility"]=compat(x)
        picked.append(y)
        if len(picked)>=limit:break
    return sorted(picked,key=lambda x:x["date"])

def _nearest_session_index(df,target_date):
    if df.empty:return None
    dates=list(df["d"])
    return min(range(len(dates)),key=lambda i:abs((dates[i]-target_date).days))

def evaluate_historical_forecast(df,forecast,wide_sessions=10,price_tol_pct=3.0,partial_price_tol_pct=6.0,
                                 time_tol_sessions=3,partial_time_tol_sessions=7):
    d=forecast.get("date");target=float(forecast.get("price") or 0)
    if not d or target<=0 or df.empty:
        return {**forecast,"status":"غير قابل للتقييم","status_key":"na","status_color":"#64748b","actual_price":None,"actual_date":None,"price_error_pct":None,"time_error_sessions":None}
    center=_nearest_session_index(df,d)
    if center is None:
        return {**forecast,"status":"غير قابل للتقييم","status_key":"na","status_color":"#64748b","actual_price":None,"actual_date":None,"price_error_pct":None,"time_error_sessions":None}
    lo=max(0,center-wide_sessions);hi=min(len(df)-1,center+wide_sessions);w=df.iloc[lo:hi+1]
    if w.empty:
        return {**forecast,"status":"غير قابل للتقييم","status_key":"na","status_color":"#64748b","actual_price":None,"actual_date":None,"price_error_pct":None,"time_error_sessions":None}
    if forecast.get("type")=="TOP":
        rel=int(np.nanargmax(w["h"].to_numpy(float)));actual_price=float(w.iloc[rel]["h"])
    else:
        rel=int(np.nanargmin(w["l"].to_numpy(float)));actual_price=float(w.iloc[rel]["l"])
    actual_i=lo+rel;actual_date=df.iloc[actual_i]["d"]
    price_error=abs(actual_price/target-1)*100;time_error=abs(actual_i-center)
    if price_error<=price_tol_pct and time_error<=time_tol_sessions:status,status_key="تحقق","hit"
    elif ((price_error<=price_tol_pct and time_error<=partial_time_tol_sessions) or (price_error<=partial_price_tol_pct and time_error<=time_tol_sessions)):status,status_key="تحقق جزئي","partial"
    else:status,status_key="لم يتحقق","miss"
    status_color={"hit":"#15803d","partial":"#d97706","miss":"#b91c1c","na":"#64748b"}[status_key]
    return {**forecast,"status":status,"status_key":status_key,"status_color":status_color,"actual_price":round(actual_price,4),"actual_date":actual_date,"price_error_pct":round(price_error,2),"time_error_sessions":int(time_error)}

def _historical_expected_turns(df,pivots,chart_start=None,wing=7):
    """As-of historical forecasts; each anchor stops when the next anchor is available."""
    if not pivots:return [],[]
    today=date.today();hist_tops=[];hist_lows=[]
    subset=pivots[-14:];base_idx=max(0,len(pivots)-len(subset))
    for local_idx,p in enumerate(subset):
        anchor_date=p["date"];anchor_price=float(p["price"]);pivot_i=int(p.get("i",0))
        confirm_i=min(len(df)-1,max(0,pivot_i+wing))
        available_date=p.get("available_date") or df.iloc[confirm_i]["d"]
        global_idx=base_idx+local_idx
        next_available=today
        if global_idx+1<len(pivots):
            npiv=pivots[global_idx+1]
            next_available=npiv.get("available_date") or npiv.get("date") or today
        if next_available<=available_date:continue

        price_items=_sq9_levels(anchor_price,p["kind"],anchor_date)+_master_price_levels(p)
        if global_idx>0:
            prev=pivots[global_idx-1]
            rng={"low":min(float(prev["price"]),anchor_price),"high":max(float(prev["price"]),anchor_price),"range":abs(anchor_price-float(prev["price"])),"anchor_date":anchor_date}
            price_items+=_range_ratio_levels(rng)
        atr_hist=_atr(df.iloc[:confirm_i+1])
        temp_pc=cluster_price_levels(price_items,anchor_price,atr_hist)
        side="resistance" if p["kind"]=="L" else "support"
        main=select_main_price_levels(temp_pc,anchor_price,side,limit=1,atr=atr_hist)
        if not main:continue
        price_pick=main[0]

        # Historical time geometry must be generated as-of the old anchor.
        # Do not use the future-only helper functions here because they filter
        # against today's date and would erase old forecast windows.
        time_items=[];unit=anchor_price
        hist_method="Gann Square Low" if p["kind"]=="L" else "Gann Square High"
        if 0<unit<=5000:
            for cycle in range(0,6):
                for frac,label,_ in IMPORTANT_RATIOS:
                    off=(cycle+frac)*unit
                    if off<=0 or off>2500:continue
                    for mode in ("calendar","trading"):
                        d=_calendar_add(anchor_date,off) if mode=="calendar" else _business_add(anchor_date,int(round(off)))
                        if d>available_date and d<next_available and d<today:
                            time_items.append({"method":hist_method,"submethod":f"{label} · {'تقويمي' if mode=='calendar' else 'جلسات'}","date":d,"strength":_time_strength(label),"mode":mode,"anchor_kind":p["kind"]})
        for cycle in range(0,6):
            for n,strength in MASTER_144:
                off=cycle*144+n
                for mode in ("calendar","trading"):
                    d=_calendar_add(anchor_date,off) if mode=="calendar" else _business_add(anchor_date,off)
                    if d>available_date and d<next_available and d<today:
                        time_items.append({"method":"Master 144","submethod":f"{n} · {'تقويمي' if mode=='calendar' else 'جلسات'}","date":d,"strength":strength,"mode":mode,"anchor_kind":p["kind"]})
        for n in MIKULA_225_CELL_COUNTS:
            d=_business_add(anchor_date,n)
            if d>available_date and d<next_available and d<today:
                time_items.append({"method":"Mikula 225° Cells","submethod":f"{n} bars","date":d,"strength":min(96,78+int(math.log(max(n,9),3))*3),"mode":"trading","anchor_kind":p["kind"]})
        strong_times=[x for x in cluster_time_dates(time_items) if x["strength"]>=75]
        strong_times=sorted(strong_times,key=lambda x:(-x["strength"],x["date"]))[:2]
        for t in strong_times:
            st=int(round(min(100,.62*price_pick["strength"]+.38*t["strength"]+5)))
            item={"type":"TOP" if p["kind"]=="L" else "LOW","price":round(float(price_pick["level"]),4),"date":t["date"],"strength":st,"methods":sorted(set(price_pick["methods"]+t["methods"])),"color":strength_color(st),"anchor_date":anchor_date,"anchor_price":round(anchor_price,4),"available_date":available_date}
            (hist_tops if p["kind"]=="L" else hist_lows).append(item)

    def compact(items):
        items=sorted(items,key=lambda x:(x["date"],-x["strength"]));out=[]
        for it in items:
            if out and abs((it["date"]-out[-1]["date"]).days)<=2:
                if it["strength"]>out[-1]["strength"]:out[-1]=it
            else:out.append(it)
        return out[-12:]
    return [evaluate_historical_forecast(df,x) for x in compact(hist_tops)],[evaluate_historical_forecast(df,x) for x in compact(hist_lows)]

def _backtest_stats(tops,lows):
    xs=[x for x in tops+lows if x.get("status_key") in ("hit","partial","miss")]
    n=len(xs);hits=sum(x["status_key"]=="hit" for x in xs);partials=sum(x["status_key"]=="partial" for x in xs);misses=sum(x["status_key"]=="miss" for x in xs)
    if n:
        raw=(100*hits+60*partials)/n
        reliability=50+(raw-50)*(n/(n+4))
        pe=[x["price_error_pct"] for x in xs if x.get("price_error_pct") is not None]
        te=[x["time_error_sessions"] for x in xs if x.get("time_error_sessions") is not None]
        med_price=float(np.median(pe)) if pe else None;med_time=float(np.median(te)) if te else None
    else:
        raw=50;reliability=50;med_price=med_time=None
    return {"samples":n,"hits":hits,"partials":partials,"misses":misses,"hit_rate":round(100*hits/n,1) if n else None,"useful_rate":round(100*(hits+partials)/n,1) if n else None,"raw_score":round(raw,1),"reliability_score":round(reliability,1),"median_price_error_pct":round(med_price,2) if med_price is not None else None,"median_time_error_sessions":round(med_time,1) if med_time is not None else None,"sample_quality":"كافٍ" if n>=FINAL_BACKTEST_MIN_SAMPLES else "محدود"}

def _anchor_strength(market,low,high):
    if market=="EGX":
        vals=[float(x.get("stability_pct",0)) for x in (low,high) if x]
        return round(sum(vals)/len(vals),1) if vals else 50.0
    return 65.0

def _final_decision_score(geometry,anchor_strength,backtest):
    n=int(backtest.get("samples",0));rel=float(backtest.get("reliability_score",50))
    if n>=FINAL_BACKTEST_MIN_SAMPLES:score=.55*geometry+.25*anchor_strength+.20*rel
    else:score=.70*geometry+.30*anchor_strength
    return int(round(max(0,min(100,score))))

def _pair_turns(pc,tc,current,atr,anchor_strength,backtest):
    sups=select_main_price_levels(pc,current,"support",limit=2,atr=atr)
    ress=select_main_price_levels(pc,current,"resistance",limit=2,atr=atr)
    top_times=select_final_time_windows(tc,"TOP",limit=2)
    low_times=select_final_time_windows(tc,"LOW",limit=2)
    def make(levels,times,typ):
        ans=[]
        for i,p in enumerate(levels):
            t=times[i] if i<len(times) else (times[-1] if times else None)
            geometry=int(round(p["strength"] if not t else min(100,.62*p["strength"]+.38*t["strength"]+5)))
            decision=_final_decision_score(geometry,anchor_strength,backtest)
            ans.append({"type":typ,"price":p["level"],"date":t["date"] if t else None,"window_start":t.get("window_start") if t else None,"window_end":t.get("window_end") if t else None,"strength":decision,"decision_score":decision,"geometry_strength":geometry,"price_strength":p["strength"],"time_strength":t["strength"] if t else None,"methods":sorted(set(p["methods"]+(t["methods"] if t else []))),"color":strength_color(decision),"level_count":p.get("count",1),"distance_pct":p.get("distance_pct"),"anchor_kinds":p.get("anchor_kinds",[])})
        return ans
    return make(ress,top_times,"TOP"),make(sups,low_times,"LOW"),top_times,low_times

def _decision_summary(last,ns,nr,next_times,backtest,anchor_strength):
    sup_dist=abs(100*(last/ns["level"]-1)) if ns and ns.get("level") else None
    res_dist=abs(100*(nr["level"]/last-1)) if nr and nr.get("level") else None
    if sup_dist is not None and res_dist is not None:
        total=sup_dist+res_dist;position=(100*sup_dist/total) if total else 50
        if position>=70:zone="قريب من المقاومة الرئيسية"
        elif position<=30:zone="قريب من الدعم الرئيسي"
        else:zone="منتصف النطاق الرئيسي"
    elif nr:zone="أقرب لمقاومة رئيسية";position=None
    elif ns:zone="أقرب لدعم رئيسي";position=None
    else:zone="لا توجد مستويات رئيسية كافية";position=None
    return {"zone":zone,"position_pct":round(position,1) if position is not None else None,"support_distance_pct":round(sup_dist,2) if sup_dist is not None else None,"resistance_distance_pct":round(res_dist,2) if res_dist is not None else None,"next_window":next_times[0] if next_times else None,"backtest_reliability":backtest.get("reliability_score"),"anchor_strength":anchor_strength}

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
    market=symbol_market(symbol)
    effective_limit=max(int(limit or 900),FINAL_HISTORY_LIMIT_EGX if market=="EGX" else FINAL_HISTORY_LIMIT_US)
    df=load_daily(DB,symbol,effective_limit)
    if df.empty:return None

    market_pivots=[];historical_market_pivots=[]
    if market=="EGX":
        market_pivots=get_egx_market_pivots(DB)
        if len(market_pivots)<2:raise RuntimeError("لا توجد قمم/قيعان سوق رئيسية كافية.")
        all_market_stock_pivots=_map_market_pivots_to_stock(df,market_pivots)
        if len(all_market_stock_pivots)<2:raise RuntimeError("لا توجد بيانات كافية للسهم حول مرتكزات السوق الرئيسية.")
        last_low=next((x for x in reversed(all_market_stock_pivots) if x["kind"]=="L"),None)
        last_high=next((x for x in reversed(all_market_stock_pivots) if x["kind"]=="H"),None)
        piv=[x for x in (last_low,last_high) if x is not None];piv.sort(key=lambda x:x["i"])
        historical_market_pivots=all_market_stock_pivots
        anchor_source="EGX_MARKET_CONSENSUS_FINAL"
    else:
        piv=detect_pivots(df,wing=7,min_move_pct=4.0)
        historical_market_pivots=piv
        anchor_source="LOCAL_CONFIRMED"

    low,high=_latest_pair(piv);rng=_last_completed_range(piv);last=float(df.iloc[-1]["c"]);atr=_atr(df)
    if not low or not high:raise RuntimeError("لا يوجد قاع وقمة مؤكدان كافيان لبناء محرك جان.")

    pi=[]
    pi+=_sq9_levels(low["price"],"L",low["date"])+_master_price_levels(low)
    pi+=_sq9_levels(high["price"],"H",high["date"])+_master_price_levels(high)
    pi+=_range_ratio_levels(rng)
    pc=cluster_price_levels(pi,last,atr)

    ti=[]
    ti+=_tag_times(_repeat_square_dates(low["date"],low["price"],"Gann Square Low"),"L")
    ti+=_tag_times(_master_144_dates(low["date"]),"L")+_tag_times(_mikula_dates(low["date"]),"L")
    ti+=_tag_times(_repeat_square_dates(high["date"],high["price"],"Gann Square High"),"H")
    ti+=_tag_times(_master_144_dates(high["date"]),"H")+_tag_times(_mikula_dates(high["date"]),"H")
    if rng and rng["range"]>0:ti+=_tag_times(_repeat_square_dates(rng["anchor_date"],rng["range"],"Gann Square Range"),"RANGE")
    tc=cluster_time_dates(ti)

    previous_tops,previous_lows=_historical_expected_turns(df,historical_market_pivots,df.iloc[0]["d"],wing=7)
    backtest=_backtest_stats(previous_tops,previous_lows)
    anchor_strength=_anchor_strength(market,low,high)

    tops,lows,top_times,low_times=_pair_turns(pc,tc,last,atr,anchor_strength,backtest)
    all_final_times=sorted({x["date"]:x for x in top_times+low_times}.values(),key=lambda x:x["date"])
    next_times=all_final_times[:2]

    main_sups=select_main_price_levels(pc,last,"support",limit=2,atr=atr)
    main_ress=select_main_price_levels(pc,last,"resistance",limit=2,atr=atr)
    ns=main_sups[0] if main_sups else None;nr=main_ress[0] if main_ress else None
    decision_summary=_decision_summary(last,ns,nr,next_times,backtest,anchor_strength)
    scores=[x["decision_score"] for x in tops+lows]
    overall=max(scores) if scores else int(round(anchor_strength))

    candles=df.tail(220)
    return {"engine_version":GANN_ENGINE_VERSION,"symbol":symbol.upper(),"market":market,"anchor_source":anchor_source,"market_pivots":market_pivots,"stock_market_anchors":historical_market_pivots[-6:] if market=="EGX" else [],"last_price":round(last,4),"last_date":df.iloc[-1]["d"],"atr14":round(atr,4),"latest_low":low,"latest_high":high,"range":rng,"price_clusters":pc,"time_clusters":tc,"next_times":next_times,"top_times":top_times,"low_times":low_times,"nearest_support":ns,"nearest_resistance":nr,"tops":tops,"lows":lows,"previous_tops":previous_tops,"previous_lows":previous_lows,"backtest":backtest,"anchor_strength":anchor_strength,"decision_summary":decision_summary,"overall_strength":overall,"overall_color":strength_color(overall),"chart":{"dates":[d.isoformat() for d in candles["d"]],"open":[round(float(x),4) for x in candles["o"]],"high":[round(float(x),4) for x in candles["h"]],"low":[round(float(x),4) for x in candles["l"]],"close":[round(float(x),4) for x in candles["c"]]},"method_rows":build_method_rows(last,pc,tc)}

def market_page(DB,query="",page=1,per_page=100,market="US"):
    market=normalize_market(market)
    symbols,total,page,pages=list_symbols(DB,query,page,per_page,market=market); rows=[]
    for sym in symbols:
        try:
            a=analyze_symbol(DB,sym,420)
            if not a: continue
            rows.append({"symbol":sym,"last_price":a["last_price"],"last_date":a["last_date"],"support":a["nearest_support"],"resistance":a["nearest_resistance"],"next_time":a["next_times"][0] if a["next_times"] else None,"top1":a["tops"][0] if a["tops"] else None,"top2":a["tops"][1] if len(a["tops"])>1 else None,"low1":a["lows"][0] if a["lows"] else None,"low2":a["lows"][1] if len(a["lows"])>1 else None,"strength":a["overall_strength"],"color":a["overall_color"],"zone":a["decision_summary"]["zone"],"backtest":a["backtest"]})
        except Exception as e:
            rows.append({"symbol":sym,"error":str(e)[:180],"strength":0,"color":"#64748b"})
    return {"rows":rows,"total":total,"page":page,"pages":pages,"query":query or "","market":market}

def source_methodology():
    return [
      {"name":"Gann: Square Range / Low / High","rule":"مساواة عدد نقاط السعر بعدد فترات الزمن مع نسب 1/4،1/3،1/2،2/3،3/4 والمربع الكامل.","source":"W.D. Gann Master Commodities Course — Chapter 6."},
      {"name":"Gann Master Square of 144","rule":"نقاط وتقاطع 36،45،48،54،63،72،81،90،96،108،117،126،135،144.","source":"W.D. Gann Master Mathematical Price, Time and Trend Calculator."},
      {"name":"Bowden","rule":"Squaring a Low / High / Range مع calendar days وtrading days وضبط 45° كـ1×1 حقيقي.","source":"David E. Bowden — Squaring Time and Price."},
      {"name":"Mikula Square of Nine","rule":"(sqrt(P) ± angle/180)^2 ومستويات Cardinal/Fixed Cross وتواريخ Cell counts.","source":"Patrick Mikula — The Definitive Guide to Forecasting Using W.D. Gann's Square of Nine."},
      {"name":"Final Decision Layer","rule":"يجمع الالتقاء الهندسي + ثبات الـAnchor + Backtest تاريخي as-of. درجة القرار ليست احتمال نجاح ولا توصية شراء/بيع.","source":"طبقة قرار داخلية V5.0 فوق طرق جان/Mikula."}
    ]
