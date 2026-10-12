#!/usr/bin/env python3
"""Analyze 11 pre-burst indicators at first_entry_date of EGX +500% episodes.
Run: python egx_500_entry_indicators.py --episodes /tmp/egx-500-24m-egypt/500pct_episodes.csv --output-dir /tmp/egx-500-entry-indicators
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

FEATURES = ["rsi14", "momentum5_pct", "momentum20_pct", "close_vs_sma20_pct",
            "close_vs_sma50_pct", "bbz20", "atr14_pct", "return_20",
            "position_20_pct", "volume5_vs_50", "repeat_count_1pct_60"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/tmp/egx-500-24m-egypt/500pct_episodes.csv")
    ap.add_argument("--output-dir", default="/tmp/egx-500-entry-indicators")
    ap.add_argument("--start", default="2022-10-12", help="inclusive first_hit_date")
    ap.add_argument("--end", default="2026-10-12", help="exclusive first_hit_date")
    a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    episodes_path=Path(a.episodes)
    if not episodes_path.exists():
        raise SystemExit(f"Missing episodes file: {episodes_path}. Re-run Egypt-only +500% analysis in same Railway instance.")
    eps=pd.read_csv(episodes_path,dtype={"symbol":str})
    required={"symbol","first_entry_date","first_hit_date"}
    if not required.issubset(eps.columns):
        raise SystemExit(f"Missing episode columns: {sorted(required-set(eps.columns))}; found {list(eps.columns)}")
    for c in ["first_entry_date","first_hit_date"]:
        eps[c]=pd.to_datetime(eps[c],errors="coerce").dt.normalize()
    eps["symbol"]=eps["symbol"].astype(str).str.strip().str.upper()
    eps=eps.loc[eps.symbol.str.endswith(".CA") & eps.first_hit_date.ge(pd.Timestamp(a.start)) & eps.first_hit_date.lt(pd.Timestamp(a.end))].copy()
    eps=eps.dropna(subset=["first_entry_date","first_hit_date"])
    eps=eps.sort_values(["symbol","first_entry_date","first_hit_date"]).reset_index(drop=True)
    eps.insert(0,"episode_id",range(1,len(eps)+1))
    if eps.empty: raise SystemExit("No episodes after filtering; verify start/end.")
    print(f"EPISODES: {len(eps)}; DISTINCT STOCKS: {eps.symbol.nunique()}",flush=True)

    # Fetch strictly the indicator row at the entry date, never after entry.
    # A lateral join returns a single row; any duplicate indicator rows are counted separately below.
    args={}
    values=[]
    for i,r in eps.iterrows():
        args[f"s{i}"]=r["symbol"]
        args[f"d{i}"]=r["first_entry_date"].date()
        values.append(f"(:s{i}, CAST(:d{i} AS date), {int(r['episode_id'])})")
    q=f"""
    WITH wanted(symbol, entry_date, episode_id) AS (VALUES {', '.join(values)})
    SELECT w.episode_id, i.symbol AS indicator_symbol, i.reference_utc AS indicator_reference_utc,
           {', '.join('i.'+c for c in FEATURES)}
    FROM wanted w
    LEFT JOIN LATERAL (
      SELECT symbol, reference_utc, {', '.join(FEATURES)}
      FROM us_preburst_full_history_v5 i
      WHERE lower(i.market) = 'egypt'
        AND i.symbol = w.symbol
        AND CAST(i.reference_utc AS date) = w.entry_date
      ORDER BY i.reference_utc DESC
      LIMIT 1
    ) i ON TRUE
    ORDER BY w.episode_id
    """
    engine=create_engine(os.environ["DATABASE_URL"],pool_pre_ping=True)
    with engine.connect() as conn:
        matched=pd.read_sql_query(text(q), conn, params=args)
    # No indicator dates are inferred or filled with future values.
    joined=eps.merge(matched,on="episode_id",how="left",validate="one_to_one")
    joined["indicator_exact_date_match"]=joined["indicator_symbol"].notna()
    joined["indicators_complete_11"]=joined["indicator_exact_date_match"] & joined[FEATURES].notna().all(axis=1)
    joined["missing_feature_count"]=joined[FEATURES].isna().sum(axis=1)
    joined.to_csv(out/"500pct_episodes_with_entry_indicators.csv",index=False,encoding="utf-8-sig")
    missing=joined.loc[~joined.indicators_complete_11, ["symbol","first_entry_date","first_hit_date","indicator_exact_date_match","missing_feature_count"]]
    missing.to_csv(out/"missing_or_incomplete_indicators.csv",index=False,encoding="utf-8-sig")

    full=joined.loc[joined.indicators_complete_11].copy()
    if len(full):
        summary=full[FEATURES].describe(percentiles=[.10,.25,.50,.75,.90]).T
        summary.index.name="indicator"
        summary.to_csv(out/"entry_indicators_distribution.csv",encoding="utf-8-sig")
        summary[["count","mean","std","min","10%","25%","50%","75%","90%","max"]].round(4).to_csv(out/"entry_indicators_compact_stats.csv",encoding="utf-8-sig")
        # One observation per stock: first eligible episode by entry date.
        unique=full.sort_values(["symbol","first_entry_date"]).drop_duplicates("symbol")
        unique.to_csv(out/"one_first_episode_per_stock.csv",index=False,encoding="utf-8-sig")
        unique[FEATURES].describe(percentiles=[.1,.25,.5,.75,.9]).T.to_csv(out/"one_per_stock_statistics.csv",encoding="utf-8-sig")
        for f in FEATURES:
            vals=pd.to_numeric(full[f],errors="coerce")
            bins=10 if vals.nunique()>10 else max(1,vals.nunique())
            cuts=pd.cut(vals,bins=bins,include_lowest=True,duplicates="drop")
            # Useful descriptive frequencies; not conditional win rates.
            tab=cuts.value_counts(sort=False).rename_axis("range").reset_index(name="episode_count")
            tab.to_csv(out/("hist_"+f+".csv"),index=False,encoding="utf-8-sig")
    s={"definition":"EGX .CA +500% (6x) within 24 months, first_entry_date indicator snapshot",
       "hit_date_window_inclusive_start":a.start,"hit_date_window_exclusive_end":a.end,
       "source_episodes":str(episodes_path),"episodes_in_window":int(len(eps)),
       "distinct_stocks":int(eps.symbol.nunique()),
       "exact_entry_day_indicator_matches":int(joined.indicator_exact_date_match.sum()),
       "complete_11_feature_episodes":int(joined.indicators_complete_11.sum()),
       "complete_11_feature_distinct_stocks":int(full.symbol.nunique()),
       "incomplete_or_unmatched_episodes":int((~joined.indicators_complete_11).sum()),
       "features":FEATURES,
       "cautions":["Dates selected retrospectively using eventual +500% hit, so indicator distributions are descriptive, not predictive.",
                   "Episodes are not independent stocks; use one_first_episode_per_stock.csv for stock-weighted statistics.",
                   "A first-entry date chosen as earliest successful entry can be mechanically biased and is not necessarily an actionable signal.",
                   "Exact entry-day joins only; missing features not forward filled.",
                   "OHLC High data may be unadjusted for corporate actions; verify extraordinary returns."]}
    (out/"summary.json").write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(s,indent=2,ensure_ascii=False),flush=True)
    if len(full):
        print("\n11 INDICATOR STATS FOR MATCHED EPISODES:",flush=True)
        print(full[FEATURES].describe().round(3).to_string(),flush=True)
    print("OUTPUT:",out,flush=True)

if __name__=="__main__": main()
