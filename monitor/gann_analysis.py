
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

GANN_ENGINE_VERSION="7.0 CONSTANCE BROWN PUBLIC-METHOD"
FINAL_TIME_HORIZON_DAYS=270
FINAL_TIME_WINDOW_DAYS=2
FINAL_TIME_MIN_SEPARATION_DAYS=7
FINAL_LEVEL_MIN_SEPARATION_PCT=1.75
FINAL_BACKTEST_MIN_SAMPLES=3
FINAL_HISTORY_LIMIT_EGX=2200
FINAL_HISTORY_LIMIT_US=1400
FINAL_VALIDATION_MIN_SAMPLES=6
FINAL_METHOD_MIN_SAMPLES=3
FINAL_RELIABILITY_GATE=45.0
FINAL_USEFUL_RATE_GATE=35.0
FINAL_METHOD_RELIABILITY_GATE=42.0
FINAL_DIRECTION_MIN_SCORE=55.0
FINAL_MAX_DIRECTION_LEVEL_DISTANCE_PCT=35.0
FINAL_SPLIT_RATIO=0.60

# Constance Brown public-method implementation.
# Public sources document the Gann Wheel angles, three-axis confluence,
# square-bar-count time work, fixed-scale diagonal work, and Composite Index.
BROWN_WHEEL_ANGLES=[45,90,120,180,240,270,315,360]
BROWN_ANGLE_STRENGTH={45:82,90:92,120:94,180:98,240:94,270:92,315:82,360:99}
BROWN_WHEEL_ROTATIONS=3
BROWN_PRICE_CLUSTER_PCT=0.006
BROWN_TIME_CLUSTER_DAYS=2
BROWN_TIME_HORIZON_DAYS=270
BROWN_SQUARE_BAR_COUNTS=[9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361]
BROWN_SWING_TIME_FACTORS=[0.75,1.0,1.25,1.5]
BROWN_MIN_HORIZONTAL_ORIGINS=2
BROWN_MIN_VERTICAL_ORIGINS=2
BROWN_DIAGONAL_TOL_ATR=1.25
BROWN_NOISE_WINDOW_DAYS=30
BROWN_NOISE_MAX_CLUSTERS=5
BROWN_CI_FAST=13
BROWN_CI_SLOW=33

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


def _brown_rsi(series,n=14):
    s=pd.Series(series,dtype=float)
    d=s.diff()
    up=d.clip(lower=0.0)
    dn=(-d.clip(upper=0.0))
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=au/ad.replace(0,np.nan)
    rsi=100-(100/(1+rs))
    return rsi.fillna(50.0)

def _brown_composite_index(df):
    """Public Constance Brown Composite Index:
    Momentum(9) of RSI(14) + SMA(3) of RSI(3), with 13/33 SMA signal lines.
    """
    c=pd.Series(df["c"].astype(float).values,index=df.index)
    r14=_brown_rsi(c,14)
    r3=_brown_rsi(c,3)
    ci=(r14-r14.shift(9))+r3.rolling(3,min_periods=1).mean()
    fast=ci.rolling(BROWN_CI_FAST,min_periods=1).mean()
    slow=ci.rolling(BROWN_CI_SLOW,min_periods=1).mean()
    return pd.DataFrame({"rsi14":r14,"ci":ci,"ci_fast":fast,"ci_slow":slow},index=df.index)

def _brown_oscillator_confirmation(df,pivots):
    """Conservative directional confirmation using Brown's Composite Index.
    Returns warnings/confirmation, never creates a Gann target by itself.
    """
    o=_brown_composite_index(df)
    if o.empty:
        return {"bullish":False,"bearish":False,"state":"NEUTRAL","reason":"no oscillator data"}
    last=o.iloc[-1]
    bullish_mom=bool(last["ci"]>last["ci_fast"] and last["ci_fast"]>=last["ci_slow"])
    bearish_mom=bool(last["ci"]<last["ci_fast"] and last["ci_fast"]<=last["ci_slow"])

    lows=[x for x in pivots if x["kind"]=="L" and 0<=int(x["i"])<len(o)]
    highs=[x for x in pivots if x["kind"]=="H" and 0<=int(x["i"])<len(o)]
    bull_div=False;bear_div=False
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        ca=float(o.iloc[int(a["i"])]["ci"]);cb=float(o.iloc[int(b["i"])]["ci"])
        bull_div=(float(b["price"])<float(a["price"]) and cb>ca)
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        ca=float(o.iloc[int(a["i"])]["ci"]);cb=float(o.iloc[int(b["i"])]["ci"])
        bear_div=(float(b["price"])>float(a["price"]) and cb<ca)

    bullish=bool(bull_div or bullish_mom)
    bearish=bool(bear_div or bearish_mom)
    state="BULLISH" if bullish and not bearish else "BEARISH" if bearish and not bullish else "MIXED"
    return {
        "bullish":bullish,"bearish":bearish,"bull_divergence":bull_div,"bear_divergence":bear_div,
        "bullish_momentum":bullish_mom,"bearish_momentum":bearish_mom,"state":state,
        "ci":round(float(last["ci"]),2),"ci_fast":round(float(last["ci_fast"]),2),
        "ci_slow":round(float(last["ci_slow"]),2),"rsi14":round(float(last["rsi14"]),2)
    }

def _brown_wheel_levels(anchor,primary_only=False):
    """Constance Brown/Gann Wheel price objectives using public square-root factors.
    360° = +/-2 in sqrt(price); factor = angle/180.
    """
    if not anchor:return []
    px=float(anchor["price"]);root=math.sqrt(max(px,1e-12));out=[]
    kind=anchor.get("kind")
    for rot in range(BROWN_WHEEL_ROTATIONS):
        for angle in BROWN_WHEEL_ANGLES:
            factor=(angle/180.0)+(2.0*rot)
            strength=max(65,BROWN_ANGLE_STRENGTH.get(angle,80)-rot*7)
            dirs=["up","down"]
            if primary_only:
                dirs=["up"] if kind=="L" else ["down"]
            for direction in dirs:
                rr=root+factor if direction=="up" else root-factor
                if rr<=0:continue
                lvl=rr*rr
                out.append({
                    "method":"Brown Gann Wheel",
                    "submethod":f"{angle}° {'up' if direction=='up' else 'down'} · rotation {rot}",
                    "level":lvl,"strength":strength,
                    "pivot_kind":kind,"pivot_date":anchor["date"],
                    "origin":f'{kind}:{anchor["date"]}',"angle":angle,"rotation":rot,
                    "direction":direction,
                })
    return out

def _brown_cluster_price(items,current,atr):
    items=sorted([x for x in items if _safe_float(x.get("level")) and x["level"]>0],key=lambda x:x["level"])
    if not items:return []
    tol=max(current*BROWN_PRICE_CLUSTER_PCT,atr*0.35 if atr else 0.0)
    cs=[]
    for it in items:
        if not cs or abs(it["level"]-cs[-1]["level"])>tol:
            cs.append({"items":[it],"level":float(it["level"])})
        else:
            c=cs[-1];c["items"].append(it)
            w=np.array([max(1,float(q.get("strength",1))) for q in c["items"]])
            v=np.array([float(q["level"]) for q in c["items"]])
            c["level"]=float(np.average(v,weights=w))
    out=[]
    for c in cs:
        origins=sorted(set(x.get("origin") for x in c["items"] if x.get("origin")))
        angles=sorted(set(int(x["angle"]) for x in c["items"] if x.get("angle") is not None))
        base=max(float(x["strength"]) for x in c["items"])
        confluence_bonus=min(22,8*max(0,len(origins)-1)+2*max(0,len(angles)-1))
        st=int(round(min(100,base+confluence_bonus)))
        out.append({
            "level":round(c["level"],4),"strength":st,
            "methods":["Brown Gann Wheel"],
            "labels":[x["submethod"] for x in c["items"]],
            "origins":origins,"origin_count":len(origins),"angles":angles,
            "count":len(c["items"]),
            "side":"resistance" if c["level"]>current else "support",
            "distance_pct":round(100*(c["level"]/current-1),2),
            "horizontal_confluence":len(origins)>=BROWN_MIN_HORIZONTAL_ORIGINS,
        })
    return out

def _brown_square_bar_dates(anchor):
    if not anchor:return []
    out=[];today=date.today()
    for n in BROWN_SQUARE_BAR_COUNTS:
        d=_business_add(anchor["date"],n)
        if d>=today-timedelta(days=10) and d<=today+timedelta(days=BROWN_TIME_HORIZON_DAYS):
            out.append({
                "method":"Brown Square Bar Count","submethod":f"{n} bars",
                "date":d,"strength":min(98,72+int(math.sqrt(n))*2),
                "anchor_kind":anchor.get("kind"),"origin":f'{anchor.get("kind")}:{anchor["date"]}',
                "bar_count":n,
            })
    return out

def _brown_swing_rhythm_dates(low,high,df):
    """Public-source-inspired rhythm proxy: Brown explicitly discusses expanding/contracting cycles.
    Exact proprietary Thirty-Second Jewel time equations are not public; this uses the observed
    principal swing bar count only as a secondary vertical-axis input.
    """
    if not low or not high:return []
    i1=int(low.get("i",0));i2=int(high.get("i",0))
    bars=max(1,abs(i2-i1))
    anchor=high if high["date"]>=low["date"] else low
    out=[];today=date.today()
    for f in BROWN_SWING_TIME_FACTORS:
        n=max(1,int(round(bars*f)))
        d=_business_add(anchor["date"],n)
        if d>=today-timedelta(days=10) and d<=today+timedelta(days=BROWN_TIME_HORIZON_DAYS):
            out.append({
                "method":"Brown Swing Rhythm Proxy","submethod":f"{f:.2f}× prior swing ({n} bars)",
                "date":d,"strength":78 if abs(f-1)<1e-9 else 70,
                "anchor_kind":"RANGE","origin":f'RANGE:{anchor["date"]}',"bar_count":n,
            })
    return out

def _brown_cluster_time(items,window_days=BROWN_TIME_CLUSTER_DAYS):
    if not items:return []
    items=sorted(items,key=lambda x:x["date"]);cs=[]
    for it in items:
        if not cs or abs((it["date"]-cs[-1]["date"]).days)>window_days:
            cs.append({"date":it["date"],"items":[it]})
        else:
            cs[-1]["items"].append(it)
            cs[-1]["date"]=max(cs[-1]["items"],key=lambda q:q["strength"])["date"]
    out=[]
    for c in cs:
        origins=sorted(set(x.get("origin") for x in c["items"] if x.get("origin")))
        methods=sorted(set(x.get("method") for x in c["items"] if x.get("method")))
        base=max(float(x["strength"]) for x in c["items"])
        st=int(round(min(100,base+min(22,9*max(0,len(origins)-1)+5*max(0,len(methods)-1)))))
        out.append({
            "date":c["date"],"strength":st,"methods":methods,
            "labels":[x["submethod"] for x in c["items"]],
            "origins":origins,"origin_count":len(origins),"count":len(c["items"]),
            "vertical_confluence":len(origins)>=BROWN_MIN_VERTICAL_ORIGINS,
        })
    return out

def _brown_time_noise(clusters):
    """Brown warns that a congested mess of cycle targets over a wide interval is disharmonic noise."""
    xs=sorted([x for x in clusters if x["date"]>=date.today()],key=lambda x:x["date"])
    noisy=set()
    for i,x in enumerate(xs):
        lo=x["date"]-timedelta(days=BROWN_NOISE_WINDOW_DAYS//2)
        hi=x["date"]+timedelta(days=BROWN_NOISE_WINDOW_DAYS//2)
        nearby=[q for q in xs if lo<=q["date"]<=hi]
        if len(nearby)>BROWN_NOISE_MAX_CLUSTERS:
            noisy.add(x["date"])
    out=[]
    for x in clusters:
        y=dict(x);y["disharmonic_noise"]=x["date"] in noisy
        out.append(y)
    return out

def _brown_diagonal_proxy(low,high,target_date,level,df,atr):
    """Backend proxy for Brown's diagonal/fan axis.
    Brown's public material says fixed screen scale is required; the exact proprietary
    Pythagorean/fixed-screen construction is not public. This proxy normalizes to the
    observed principal swing slope and is never allowed to create a signal alone.
    """
    if not low or not high:return {"match":False,"distance_atr":None,"note":"missing anchors"}
    a,b=(low,high) if low["date"]<=high["date"] else (high,low)
    bars=max(1,abs(int(b.get("i",0))-int(a.get("i",0))))
    base_slope=(float(b["price"])-float(a["price"]))/bars
    anchor=b
    # business-day distance approximation from anchor to target.
    try:
        future_bars=max(0,len(pd.bdate_range(pd.Timestamp(anchor["date"])+pd.offsets.BDay(1),pd.Timestamp(target_date))))
    except Exception:
        future_bars=max(0,(target_date-anchor["date"]).days)
    projected=[]
    # fan-style subdivisions around the observed 1x1 data-scale slope.
    for ratio in (0.5,1.0,2.0):
        projected.append(float(anchor["price"])+base_slope*ratio*future_bars)
    if not projected:return {"match":False,"distance_atr":None,"note":"no projection"}
    dist=min(abs(float(level)-x) for x in projected)
    atru=max(float(atr or 0),1e-9)
    da=dist/atru
    return {
        "match":bool(da<=BROWN_DIAGONAL_TOL_ATR),
        "distance_atr":round(da,2),
        "projected":[round(x,4) for x in projected],
        "note":"data-scale proxy; exact fixed-screen Brown channel is not public",
    }

def _brown_select_price_clusters(pc,current,side,limit=3):
    xs=[x for x in pc if x["side"]==side]
    # Brown principle: confluence first, then proximity.
    xs=sorted(xs,key=lambda x:(not x.get("horizontal_confluence",False),-x["origin_count"],-x["strength"],abs(x["distance_pct"])))
    picked=[]
    for x in xs:
        if abs(float(x["distance_pct"]))>50:continue
        if any(abs(x["level"]/q["level"]-1)*100<1.5 for q in picked):continue
        picked.append(x)
        if len(picked)>=limit:break
    return picked

def _brown_select_time_clusters(tc,limit=4):
    xs=[x for x in tc if x["date"]>=date.today() and x["date"]<=date.today()+timedelta(days=BROWN_TIME_HORIZON_DAYS)]
    xs=sorted(xs,key=lambda x:(not x.get("vertical_confluence",False),x.get("disharmonic_noise",False),-x["origin_count"],-x["strength"],x["date"]))
    picked=[]
    for x in xs:
        if x.get("disharmonic_noise"):continue
        if not x.get("vertical_confluence"):continue
        if any(abs((x["date"]-q["date"]).days)<7 for q in picked):continue
        y=dict(x);y["window_start"]=x["date"]-timedelta(days=2);y["window_end"]=x["date"]+timedelta(days=2)
        picked.append(y)
        if len(picked)>=limit:break
    return sorted(picked,key=lambda x:x["date"])

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
    """Historical Brown-style as-of forecasts from price/time confluence.
    Uses only anchors known at the forecast date; next pivot is evaluation boundary only.
    """
    if len(pivots)<2:return [],[]
    hist_tops=[];hist_lows=[]
    today=date.today()
    for j in range(1,len(pivots)-1):
        prev=pivots[j-1];p=pivots[j];nxt=pivots[j+1]
        available=p.get("available_date") or p.get("date")
        next_available=nxt.get("available_date") or nxt.get("date")
        if not available or not next_available or next_available<=available:continue

        known=[prev,p]
        price_items=[]
        for a in known:price_items+=_brown_wheel_levels(a,primary_only=False)
        current=float(p["price"])
        atr_hist=_atr(df.iloc[:min(len(df),max(2,int(p.get("i",1))+1))])
        pc=_brown_cluster_price(price_items,current,atr_hist)
        side="resistance" if p["kind"]=="L" else "support"
        levels=_brown_select_price_clusters(pc,current,side,limit=2)
        levels=[x for x in levels if x.get("horizontal_confluence")]
        if not levels:continue

        time_items=[]
        for a in known:time_items+=_brown_square_bar_dates(a)
        # Historical mode: rebuild dates without today's filter.
        time_items=[]
        for a in known:
            for n in BROWN_SQUARE_BAR_COUNTS:
                d=_business_add(a["date"],n)
                if d>available and d<next_available and d<today:
                    time_items.append({"method":"Brown Square Bar Count","submethod":f"{n} bars","date":d,
                                       "strength":min(98,72+int(math.sqrt(n))*2),
                                       "anchor_kind":a.get("kind"),"origin":f'{a.get("kind")}:{a["date"]}',"bar_count":n})
        # observed swing rhythm from prev->p, known as-of p
        bars=max(1,abs(int(p.get("i",0))-int(prev.get("i",0))))
        for f in BROWN_SWING_TIME_FACTORS:
            n=max(1,int(round(bars*f)));d=_business_add(p["date"],n)
            if d>available and d<next_available and d<today:
                time_items.append({"method":"Brown Swing Rhythm Proxy","submethod":f"{f:.2f}× prior swing","date":d,
                                   "strength":78 if abs(f-1)<1e-9 else 70,
                                   "anchor_kind":"RANGE","origin":f'RANGE:{p["date"]}',"bar_count":n})
        tc=_brown_time_noise(_brown_cluster_time(time_items))
        times=[x for x in tc if x.get("vertical_confluence") and not x.get("disharmonic_noise")]
        times=sorted(times,key=lambda x:(-x["strength"],x["date"]))[:2]
        if not times:continue

        typ="TOP" if p["kind"]=="L" else "LOW"
        for lvl in levels[:1]:
            for tw in times:
                diag=_brown_diagonal_proxy(prev,p,tw["date"],lvl["level"],df,atr_hist)
                methods=["Brown Gann Wheel","Brown Square Bar Count"]
                if "Brown Swing Rhythm Proxy" in tw["methods"]:methods.append("Brown Swing Rhythm Proxy")
                if diag["match"]:methods.append("Brown Diagonal Proxy")
                item={"type":typ,"price":lvl["level"],"date":tw["date"],
                      "strength":int(round(min(100,.5*lvl["strength"]+.35*tw["strength"]+(15 if diag["match"] else 0)))),
                      "methods":methods,"color":strength_color(80),
                      "anchor_date":p["date"],"anchor_price":p["price"],"available_date":available,
                      "horizontal_origins":lvl["origin_count"],"vertical_origins":tw["origin_count"],
                      "diagonal_match":diag["match"]}
                ev=evaluate_historical_forecast(df,item)
                (hist_tops if typ=="TOP" else hist_lows).append(ev)

    def compact(xs):
        xs=sorted(xs,key=lambda x:(x["date"],-x["strength"]));out=[]
        for x in xs:
            if out and abs((x["date"]-out[-1]["date"]).days)<=3:
                if x["strength"]>out[-1]["strength"]:out[-1]=x
            else:out.append(x)
        return out[-16:]
    return compact(hist_tops),compact(hist_lows)

def _outcome_points(status_key):
    if status_key=="hit":return 100.0
    if status_key=="partial":return 55.0
    if status_key=="miss":return 0.0
    return None

def _score_outcomes(xs,prior=50.0,prior_weight=6.0):
    vals=[_outcome_points(x.get("status_key")) for x in xs]
    vals=[v for v in vals if v is not None]
    n=len(vals)
    if not n:
        return {"samples":0,"hits":0,"partials":0,"misses":0,"hit_rate":None,"useful_rate":None,
                "raw_score":None,"reliability_score":prior,"median_price_error_pct":None,
                "median_time_error_sessions":None}
    hits=sum(x.get("status_key")=="hit" for x in xs)
    partials=sum(x.get("status_key")=="partial" for x in xs)
    misses=sum(x.get("status_key")=="miss" for x in xs)
    raw=sum(vals)/n
    reliability=(raw*n+prior*prior_weight)/(n+prior_weight)
    pe=[float(x["price_error_pct"]) for x in xs if x.get("price_error_pct") is not None]
    te=[float(x["time_error_sessions"]) for x in xs if x.get("time_error_sessions") is not None]
    return {
        "samples":n,"hits":hits,"partials":partials,"misses":misses,
        "hit_rate":round(100*hits/n,1),
        "useful_rate":round(100*(hits+partials)/n,1),
        "raw_score":round(raw,1),
        "reliability_score":round(reliability,1),
        "median_price_error_pct":round(float(np.median(pe)),2) if pe else None,
        "median_time_error_sessions":round(float(np.median(te)),1) if te else None,
    }

def _backtest_stats(tops,lows):
    xs=sorted([x for x in tops+lows if x.get("status_key") in ("hit","partial","miss")],
              key=lambda x:(x.get("available_date") or x.get("date"),x.get("date")))
    full=_score_outcomes(xs)
    n=len(xs)
    cut=max(1,min(n-1,int(round(n*FINAL_SPLIT_RATIO)))) if n>=2 else n
    train=xs[:cut] if n>=2 else xs
    validation=xs[cut:] if n>=2 else []
    train_stats=_score_outcomes(train)
    val_stats=_score_outcomes(validation)

    # Direction-specific diagnostics.
    top_stats=_score_outcomes([x for x in xs if x.get("type")=="TOP"])
    low_stats=_score_outcomes([x for x in xs if x.get("type")=="LOW"])

    # Validation readiness must depend on later unseen-in-time examples, not only full history.
    val_n=val_stats["samples"]
    gate_reliability=val_stats["reliability_score"] if val_n>=3 else full["reliability_score"]
    gate_useful=val_stats["useful_rate"] if val_n>=3 and val_stats["useful_rate"] is not None else full["useful_rate"]
    direction_ready=(
        full["samples"]>=FINAL_VALIDATION_MIN_SAMPLES and
        gate_reliability>=FINAL_RELIABILITY_GATE and
        (gate_useful or 0)>=FINAL_USEFUL_RATE_GATE
    )

    return {
        **full,
        "train":train_stats,
        "validation":val_stats,
        "top":top_stats,
        "low":low_stats,
        "sample_quality":"كافٍ" if full["samples"]>=FINAL_VALIDATION_MIN_SAMPLES else "محدود",
        "gate_reliability":round(float(gate_reliability),1),
        "gate_useful_rate":round(float(gate_useful or 0),1),
        "direction_ready":bool(direction_ready),
        "gate_reason":(
            "PASS" if direction_ready else
            "عينات تاريخية غير كافية" if full["samples"]<FINAL_VALIDATION_MIN_SAMPLES else
            "موثوقية الاختبار المتأخر ضعيفة" if gate_reliability<FINAL_RELIABILITY_GATE else
            "نسبة تحقق/جزئي التاريخية ضعيفة"
        )
    }

def _method_validation(previous_tops,previous_lows):
    """Per-method walk-forward evidence from as-of forecasts.

    A historical forecast can contain several converging Gann methods; each method
    receives the realized outcome of that forecast. Scores are shrunk to 50 for
    small samples, so one lucky hit cannot dominate future decisions.
    """
    xs=sorted([x for x in previous_tops+previous_lows if x.get("status_key") in ("hit","partial","miss")],
              key=lambda x:(x.get("available_date") or x.get("date"),x.get("date")))
    methods={}
    for x in xs:
        for m in sorted(set(x.get("methods") or [])):
            methods.setdefault(m,[]).append(x)
    out={}
    for m,rows in methods.items():
        st=_score_outcomes(rows,prior=50.0,prior_weight=5.0)
        n=st["samples"]
        cut=max(1,min(n-1,int(round(n*FINAL_SPLIT_RATIO)))) if n>=2 else n
        late=rows[cut:] if n>=2 else []
        late_st=_score_outcomes(late,prior=50.0,prior_weight=5.0)
        effective=late_st["reliability_score"] if late_st["samples"]>=2 else st["reliability_score"]
        validated=(n>=FINAL_METHOD_MIN_SAMPLES and effective>=FINAL_METHOD_RELIABILITY_GATE)
        out[m]={
            **st,
            "late":late_st,
            "effective_reliability":round(float(effective),1),
            "validated":bool(validated),
        }
    return out

def _method_evidence(methods,method_stats):
    vals=[];validated=[]
    for m in methods or []:
        st=method_stats.get(m)
        if not st:continue
        vals.append(float(st["effective_reliability"]))
        if st["validated"]:validated.append(m)
    if vals:
        # Average plus a small reward for independent validated methods.
        score=min(100.0,float(np.mean(vals))+min(10,2.5*max(0,len(validated)-1)))
    else:
        score=50.0
    return round(score,1),validated

def _anchor_strength(market,low,high):
    if market=="EGX":
        vals=[float(x.get("stability_pct",0)) for x in (low,high) if x]
        return round(sum(vals)/len(vals),1) if vals else 50.0
    return 65.0

def _final_decision_score(geometry,anchor_strength,backtest,method_evidence):
    """Validated ranking score; never presented as probability."""
    rel=float(backtest.get("gate_reliability",50))
    score=.35*geometry+.20*anchor_strength+.25*method_evidence+.20*rel
    return int(round(max(0,min(100,score))))

def _direction_gate(backtest,decision_score,distance_pct,validated_methods):
    if not backtest.get("direction_ready"):
        return False,backtest.get("gate_reason") or "ضعف التحقق التاريخي"
    if decision_score<FINAL_DIRECTION_MIN_SCORE:
        return False,"درجة القرار أقل من الحد المطلوب"
    if abs(float(distance_pct or 0))>FINAL_MAX_DIRECTION_LEVEL_DISTANCE_PCT:
        return False,"المستوى بعيد جدًا عن السعر الحالي"
    if not validated_methods:
        return False,"لا توجد طريقة جان اجتازت التحقق التاريخي المستقل"
    return True,"PASS"

def _pair_turns(pc,tc,current,atr,anchor_strength,backtest,method_stats):
    """Create candidates, but label TOP/LOW only after the validation gate."""
    sups=select_main_price_levels(pc,current,"support",limit=2,atr=atr)
    ress=select_main_price_levels(pc,current,"resistance",limit=2,atr=atr)
    top_times=select_final_time_windows(tc,"TOP",limit=2)
    low_times=select_final_time_windows(tc,"LOW",limit=2)

    def make(levels,times,typ):
        ans=[]
        for i,p in enumerate(levels):
            t=times[i] if i<len(times) else (times[-1] if times else None)
            methods=sorted(set(p["methods"]+(t["methods"] if t else [])))
            geometry=int(round(p["strength"] if not t else min(100,.62*p["strength"]+.38*t["strength"]+5)))
            me,validated_methods=_method_evidence(methods,method_stats)
            decision=_final_decision_score(geometry,anchor_strength,backtest,me)
            directional,reason=_direction_gate(backtest,decision,p.get("distance_pct"),validated_methods)
            state="DIRECTIONAL" if directional else "WATCH_ONLY"
            label=("قمة محتملة" if typ=="TOP" else "قاع محتمل") if directional else "نافذة مراقبة"
            ans.append({
                "type":typ if directional else "WATCH",
                "candidate_type":typ,
                "label":label,
                "state":state,
                "price":p["level"],
                "date":t["date"] if t else None,
                "window_start":t.get("window_start") if t else None,
                "window_end":t.get("window_end") if t else None,
                "strength":decision,
                "decision_score":decision,
                "geometry_strength":geometry,
                "price_strength":p["strength"],
                "time_strength":t["strength"] if t else None,
                "method_evidence":me,
                "validated_methods":validated_methods,
                "methods":methods,
                "color":strength_color(decision) if directional else "#64748b",
                "level_count":p.get("count",1),
                "distance_pct":p.get("distance_pct"),
                "anchor_kinds":p.get("anchor_kinds",[]),
                "gate_reason":reason,
            })
        return ans

    return make(ress,top_times,"TOP"),make(sups,low_times,"LOW"),top_times,low_times

def _watch_windows(tc,method_stats,limit=4):
    """Time clusters are always safe to show as watch windows, without directional claims."""
    today=date.today();end=today+timedelta(days=FINAL_TIME_HORIZON_DAYS)
    xs=[x for x in tc if today<=x["date"]<=end]
    ranked=[]
    for x in xs:
        me,valid=_method_evidence(x.get("methods") or [],method_stats)
        score=.65*float(x.get("strength",0))+.35*me
        y=dict(x)
        y["method_evidence"]=round(me,1)
        y["validated_methods"]=valid
        y["watch_score"]=int(round(max(0,min(100,score))))
        y["window_start"]=x["date"]-timedelta(days=FINAL_TIME_WINDOW_DAYS)
        y["window_end"]=x["date"]+timedelta(days=FINAL_TIME_WINDOW_DAYS)
        ranked.append(y)
    ranked.sort(key=lambda x:(-x["watch_score"],x["date"]))
    picked=[]
    for x in ranked:
        if any(abs((x["date"]-q["date"]).days)<FINAL_TIME_MIN_SEPARATION_DAYS for q in picked):
            continue
        picked.append(x)
        if len(picked)>=limit:break
    return sorted(picked,key=lambda x:x["date"])

def _decision_summary(last,ns,nr,watch_windows,backtest,anchor_strength,directional_count):
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

    if directional_count>0 and backtest.get("direction_ready"):
        regime="DIRECTIONAL"
        message="يوجد توقع اتجاهي اجتاز بوابة التحقق التاريخي."
    else:
        regime="WATCH_ONLY"
        message="جان غير موثوق اتجاهيًا على هذا السهم حاليًا؛ استخدمي النوافذ الزمنية للمراقبة فقط."

    return {
        "zone":zone,
        "position_pct":round(position,1) if position is not None else None,
        "support_distance_pct":round(sup_dist,2) if sup_dist is not None else None,
        "resistance_distance_pct":round(res_dist,2) if res_dist is not None else None,
        "next_window":watch_windows[0] if watch_windows else None,
        "backtest_reliability":backtest.get("gate_reliability"),
        "anchor_strength":anchor_strength,
        "regime":regime,
        "message":message,
        "direction_ready":bool(backtest.get("direction_ready")),
        "gate_reason":backtest.get("gate_reason"),
    }

def build_method_rows(last,pc,tc):
    methods=[
      ("Brown Gann Wheel","Square-root Gann Wheel: 45°,90°,120°,180°,240°,270°,315°,360°; horizontal price objectives."),
      ("Brown Square Bar Count","Vertical time axis from square bar counts projected from major/significant price bars."),
      ("Brown Swing Rhythm Proxy","Secondary cycle expansion/contraction proxy from the observed major swing duration."),
      ("Brown Diagonal Proxy","Data-scale fan proxy only; exact Brown fixed-screen/Pythagorean channel formula is not publicly specified."),
      ("Brown Composite Index","Momentum(9) of RSI(14) + SMA(3) of RSI(3), with 13/33 SMA confirmation."),
    ]
    rows=[]
    for method,desc in methods:
        xs=[x for x in pc if method in x.get("methods",[])]
        sup=max([x for x in xs if x["level"]<last],key=lambda x:x["level"],default=None)
        res=min([x for x in xs if x["level"]>last],key=lambda x:x["level"],default=None)
        times=[x for x in tc if method in x.get("methods",[]) and x["date"]>=date.today()][:2]
        sts=[x["strength"] for x in [sup,res] if x]+[x["strength"] for x in times]
        st=max(sts) if sts else 0
        rows.append({"method":method,"description":desc,"support":sup,"resistance":res,"times":times,
                     "strength":st,"color":strength_color(st)})
    return rows

def _brown_pair_candidates(levels,times,low,high,df,atr,osc,backtest,method_stats,current):
    out=[]
    for lvl in levels:
        typ="TOP" if lvl["side"]=="resistance" else "LOW"
        for tw in times[:2]:
            diag=_brown_diagonal_proxy(low,high,tw["date"],lvl["level"],df,atr)
            methods=["Brown Gann Wheel","Brown Square Bar Count"]
            if "Brown Swing Rhythm Proxy" in tw.get("methods",[]):methods.append("Brown Swing Rhythm Proxy")
            if diag["match"]:methods.append("Brown Diagonal Proxy")
            me,validated=_method_evidence(methods,method_stats)
            horizontal=bool(lvl.get("horizontal_confluence"))
            vertical=bool(tw.get("vertical_confluence")) and not bool(tw.get("disharmonic_noise"))
            osc_ok=osc["bearish"] if typ=="TOP" else osc["bullish"]
            three_axis=horizontal and vertical and diag["match"]
            geometry=int(round(min(100,
                .42*float(lvl["strength"])+.33*float(tw["strength"])+
                (15 if diag["match"] else 0)+(10 if osc_ok else 0)
            )))
            decision=_final_decision_score(geometry,_anchor_strength("EGX" if str(low.get("market",""))=="EGX" else "",low,high),backtest,me)
            gate_ok=bool(
                backtest.get("direction_ready") and
                three_axis and osc_ok and validated and
                decision>=FINAL_DIRECTION_MIN_SCORE and
                abs(float(lvl.get("distance_pct") or 0))<=FINAL_MAX_DIRECTION_LEVEL_DISTANCE_PCT
            )
            reason=[]
            if not backtest.get("direction_ready"):reason.append(backtest.get("gate_reason","historical gate failed"))
            if not horizontal:reason.append("no horizontal price confluence")
            if not vertical:reason.append("no clean vertical time confluence")
            if not diag["match"]:reason.append("diagonal axis not aligned")
            if not osc_ok:reason.append("Composite Index not confirming direction")
            if not validated:reason.append("no validated Brown method")
            state="DIRECTIONAL" if gate_ok else "WATCH_ONLY"
            out.append({
                "type":typ if gate_ok else "WATCH","candidate_type":typ,
                "label":("قمة محتملة" if typ=="TOP" else "قاع محتمل") if gate_ok else "منطقة سعر/زمن للمراقبة",
                "state":state,"price":lvl["level"],"date":tw["date"],
                "window_start":tw["date"]-timedelta(days=2),"window_end":tw["date"]+timedelta(days=2),
                "strength":decision,"decision_score":decision,"geometry_strength":geometry,
                "price_strength":lvl["strength"],"time_strength":tw["strength"],
                "method_evidence":me,"validated_methods":validated,"methods":methods,
                "color":strength_color(decision) if gate_ok else "#64748b",
                "distance_pct":lvl["distance_pct"],"horizontal_origins":lvl["origin_count"],
                "vertical_origins":tw["origin_count"],"diagonal":diag,
                "oscillator_confirmed":osc_ok,"three_axis_confluence":three_axis,
                "gate_reason":"PASS" if gate_ok else "; ".join(reason),
            })
    # keep the strongest unique price/time candidates
    out=sorted(out,key=lambda x:(x["state"]!="DIRECTIONAL",-x["decision_score"],abs(x["distance_pct"]),x["date"]))
    unique=[]
    for x in out:
        if any(abs(x["price"]/q["price"]-1)*100<1.0 and abs((x["date"]-q["date"]).days)<5 for q in unique):continue
        unique.append(x)
        if len(unique)>=4:break
    return unique

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
        local=detect_pivots(df,wing=7,min_move_pct=4.0)
        last_low=next((x for x in reversed(local) if x["kind"]=="L"),None)
        last_high=next((x for x in reversed(local) if x["kind"]=="H"),None)
        piv=[x for x in (last_low,last_high) if x];piv.sort(key=lambda x:x["i"])
        historical_market_pivots=local
        anchor_source="LOCAL_CONFIRMED"

    low,high=_latest_pair(piv);last=float(df.iloc[-1]["c"]);atr=_atr(df)
    if not low or not high:raise RuntimeError("لا يوجد قاع وقمة مؤكدان كافيان لمنهج Brown.")

    # Horizontal axis: Brown/Gann Wheel from both major/significant anchors.
    price_items=_brown_wheel_levels(low,primary_only=False)+_brown_wheel_levels(high,primary_only=False)
    pc=_brown_cluster_price(price_items,last,atr)
    supports=_brown_select_price_clusters(pc,last,"support",limit=3)
    resistances=_brown_select_price_clusters(pc,last,"resistance",limit=3)

    # Vertical axis: square bar counts from both anchors + secondary rhythm proxy.
    time_items=_brown_square_bar_dates(low)+_brown_square_bar_dates(high)+_brown_swing_rhythm_dates(low,high,df)
    tc=_brown_time_noise(_brown_cluster_time(time_items))
    watch_windows=_brown_select_time_clusters(tc,limit=4)

    # Brown oscillator confirmation.
    oscillator_pivots=historical_market_pivots if historical_market_pivots else detect_pivots(df,wing=7,min_move_pct=3)
    osc=_brown_oscillator_confirmation(df,oscillator_pivots)

    # Historical validation rebuilt using Brown price/time logic.
    previous_tops,previous_lows=_historical_expected_turns(df,historical_market_pivots,df.iloc[0]["d"],wing=7)
    backtest=_backtest_stats(previous_tops,previous_lows)
    method_stats=_method_validation(previous_tops,previous_lows)
    anchor_strength=_anchor_strength(market,low,high)

    levels=[x for x in resistances+supports if x.get("horizontal_confluence")]
    candidates=_brown_pair_candidates(levels,watch_windows,low,high,df,atr,osc,backtest,method_stats,last)
    directional=[x for x in candidates if x["state"]=="DIRECTIONAL"]
    tops=[x for x in candidates if x["candidate_type"]=="TOP"][:2]
    lows=[x for x in candidates if x["candidate_type"]=="LOW"][:2]

    ns=supports[0] if supports else None;nr=resistances[0] if resistances else None
    regime="DIRECTIONAL" if directional else "WATCH_ONLY"
    if regime=="DIRECTIONAL":
        message="يوجد Price-Time-Diagonal confluence مع تأكيد Composite Index واجتاز الاختبار التاريخي."
    else:
        message="منهج Brown لم يعطِ ثلاثي Confluence مؤكدًا اتجاهيًا؛ اعرضي المناطق والنوافذ للمراقبة فقط."
    decision_summary={
        "zone":"منتصف النطاق الرئيسي",
        "position_pct":None,
        "support_distance_pct":abs(ns["distance_pct"]) if ns else None,
        "resistance_distance_pct":abs(nr["distance_pct"]) if nr else None,
        "next_window":watch_windows[0] if watch_windows else None,
        "backtest_reliability":backtest.get("gate_reliability"),
        "anchor_strength":anchor_strength,"regime":regime,"message":message,
        "direction_ready":bool(directional),"gate_reason":backtest.get("gate_reason"),
        "oscillator_state":osc["state"],
    }
    if ns and nr:
        sd=abs(ns["distance_pct"]);rd=abs(nr["distance_pct"]);tot=sd+rd
        pos=100*sd/tot if tot else 50
        decision_summary["position_pct"]=round(pos,1)
        decision_summary["zone"]="قريب من المقاومة الرئيسية" if pos>=70 else "قريب من الدعم الرئيسي" if pos<=30 else "منتصف النطاق الرئيسي"

    method_rows_final=[]
    for name,st in sorted(method_stats.items(),key=lambda kv:(-kv[1]["effective_reliability"],-kv[1]["samples"],kv[0])):
        method_rows_final.append({"name":name,"samples":st["samples"],"hit_rate":st["hit_rate"],
                                  "useful_rate":st["useful_rate"],"reliability":st["effective_reliability"],
                                  "validated":st["validated"]})

    overall=max([x["decision_score"] for x in directional],default=int(round(anchor_strength)))
    candles=df.tail(220)
    return {
        "engine_version":GANN_ENGINE_VERSION,"methodology":"Constance Brown public-method implementation",
        "symbol":symbol.upper(),"market":market,"anchor_source":anchor_source,
        "market_pivots":market_pivots,"stock_market_anchors":historical_market_pivots[-6:] if market=="EGX" else [],
        "last_price":round(last,4),"last_date":df.iloc[-1]["d"],"atr14":round(atr,4),
        "latest_low":low,"latest_high":high,"range":_last_completed_range(piv),
        "price_clusters":pc,"time_clusters":tc,
        "next_times":watch_windows[:2],"watch_windows":watch_windows,
        "nearest_support":ns,"nearest_resistance":nr,
        "tops":tops,"lows":lows,"directional_signals":directional,
        "brown_candidates":candidates,"brown_oscillator":osc,
        "previous_tops":previous_tops,"previous_lows":previous_lows,
        "backtest":backtest,"method_validation":method_stats,"method_validation_rows":method_rows_final,
        "anchor_strength":anchor_strength,"decision_summary":decision_summary,
        "overall_strength":overall,"overall_color":strength_color(overall),
        "brown_public_limitations":[
            "Exact Thirty-Second Jewel fixed-screen/Pythagorean diagonal formula is not public in the sources used.",
            "Diagonal axis is a data-scale proxy and cannot create a signal by itself.",
            "Swing-rhythm expansion/contraction is a secondary proxy, not claimed as Brown's proprietary equation."
        ],
        "chart":{"dates":[d.isoformat() for d in candles["d"]],
                 "open":[round(float(x),4) for x in candles["o"]],
                 "high":[round(float(x),4) for x in candles["h"]],
                 "low":[round(float(x),4) for x in candles["l"]],
                 "close":[round(float(x),4) for x in candles["c"]]},
        "method_rows":build_method_rows(last,pc,tc),
    }

def market_page(DB,query="",page=1,per_page=100,market="US"):
    market=normalize_market(market)
    symbols,total,page,pages=list_symbols(DB,query,page,per_page,market=market); rows=[]
    for sym in symbols:
        try:
            a=analyze_symbol(DB,sym,420)
            if not a: continue
            rows.append({"symbol":sym,"last_price":a["last_price"],"last_date":a["last_date"],"support":a["nearest_support"],"resistance":a["nearest_resistance"],"next_time":a["next_times"][0] if a["next_times"] else None,"top1":a["tops"][0] if a["tops"] else None,"top2":a["tops"][1] if len(a["tops"])>1 else None,"low1":a["lows"][0] if a["lows"] else None,"low2":a["lows"][1] if len(a["lows"])>1 else None,"strength":a["overall_strength"],"color":a["overall_color"],"zone":a["decision_summary"]["zone"],"regime":a["decision_summary"]["regime"],"backtest":a["backtest"]})
        except Exception as e:
            rows.append({"symbol":sym,"error":str(e)[:180],"strength":0,"color":"#64748b"})
    return {"rows":rows,"total":total,"page":page,"pages":pages,"query":query or "","market":market}

def source_methodology():
    return [
      {"name":"Constance Brown — Gann Wheel","rule":"الأهداف السعرية من الجذر التربيعي؛ 360° = ±2 على الجذر. الزوايا المستخدمة: 45،90،120،180،240،270،315،360.","source":"Constance Brown, Technical Analysis for the Trading Professional, Ch. 9; public Gann Wheel excerpt."},
      {"name":"Brown Price/Time Confluence","rule":"السعر القوي منطقة Confluence من أكثر من projection، ثم يلتقي مع vertical time work.","source":"Constance Brown, Price and Time, Breakthroughs in Technical Analysis."},
      {"name":"Three Axes","rule":"Horizontal price + Vertical time + Diagonal axis. لا يعتمد القرار على محور واحد.","source":"Constance Brown public 2023 CMT presentation; Thirty-Second Jewel descriptions."},
      {"name":"Square Bar Count Time","rule":"الـvertical axis يعتمد bar-count time cycles من significant price bars؛ الكود يستخدم perfect-square bar counts كتنفيذ علني محافظ.","source":"Constance Brown public 2023 CMT presentation."},
      {"name":"Composite Index","rule":"Momentum(9) of RSI(14) + SMA(3) of RSI(3)، مع SMA 13/33؛ يستخدم كتأكيد وليس كصانع للهدف.","source":"Constance Brown Composite Index public formula; StockCharts ChartSchool."},
      {"name":"Public-method limitation","rule":"الـDiagonal backend proxy ليس صيغة Thirty-Second Jewel السرية؛ Brown تشترط fixed screen scale، والمعادلة الكاملة ليست منشورة في المصادر العامة المستخدمة.","source":"Optuma/Brown public materials and CMT presentation."},
    ]
