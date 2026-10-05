#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
US Outcome-Learning Optimizer V13
=================================
Built on V11's strongest area, but uses SL-risk as a VETO instead of a score penalty.

Core idea:
    rank_score = TP probability rank + Expected-return rank
    then reject only candidates whose P(SL first) is too high.

Optional market filters are tested as gates, not as permanent score weights:
    - none
    - SPY above SMA50
    - SPY above SMA200
    - stock 20d relative strength vs SPY > 0

Train: 2024 only
Validation/model selection: 2025 only
Holdout: 2026 only after the winner is chosen

Requires beside this file:
    /app/us_setup_optimizer_v11.py

Outputs:
    /data/us_v13_results.csv
    /data/us_v13_top20.csv
    /data/us_v13_report.txt
    /data/us_v13_trades_2025.csv
    /data/us_v13_trades_2026.csv
"""

import os, sys, math, argparse, subprocess, json
from pathlib import Path
from datetime import datetime, timezone

def ensure(pkg, import_name=None):
    import_name = import_name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

for pkg, imp in [
    ("numpy", None), ("pandas", None), ("scikit-learn", "sklearn"), ("requests", None)
]:
    ensure(pkg, imp)

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

try:
    import us_setup_optimizer_v11 as v11
except Exception as e:
    raise RuntimeError("ضعّي us_setup_optimizer_v11.py في /app بجانب ملف V13.") from e

VERSION = "13.1-v11-sl-veto-resume-20261005"
TRAIN_YEAR, VALID_YEAR, HOLDOUT_YEAR = 2024, 2025, 2026
OUT = Path(os.getenv("OPTIMIZER_OUT_DIR", "/data"))
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = OUT / "us_v13_results.csv"
TOP20 = OUT / "us_v13_top20.csv"
REPORT = OUT / "us_v13_report.txt"
TRADES25 = OUT / "us_v13_trades_2025.csv"
TRADES26 = OUT / "us_v13_trades_2026.csv"
CHECKPOINT_CSV = OUT / "us_v13_checkpoint_results.csv"
CHECKPOINT_JSON = OUT / "us_v13_checkpoint_state.json"

# Stay close to the V11 winner instead of re-opening a huge search.
SETUPS = [
    (18.0, 3.0, 7),
    (18.0, 3.0, 10),
    (18.0, 3.0, 12),
    (20.0, 3.0, 7),
    (20.0, 3.0, 10),
    (20.0, 3.0, 12),
    (20.0, 4.0, 10),
    (22.0, 4.0, 10),
]

SL_VETOES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
SCORE_MODES = [
    ("balanced", 0.50, 0.50),
    ("tp_heavy", 0.65, 0.35),
    ("return_heavy", 0.35, 0.65),
]
MARKET_FILTERS = ["none", "spy_sma50", "spy_sma200", "rs20_positive"]

def yahoo_spy():
    print("[SPY] downloading daily SPY from Yahoo...", flush=True)
    start = int(datetime(2022,1,1,tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp()) + 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    params = {
        "period1": start, "period2": end, "interval": "1d",
        "events": "history", "includeAdjustedClose": "true"
    }
    r = requests.get(url, params=params, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    obj = r.json()
    result = (obj.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError("Yahoo returned no SPY result.")
    item = result[0]
    ts = item.get("timestamp") or []
    ind = item.get("indicators") or {}
    adj_items = ind.get("adjclose") or []
    quote_items = ind.get("quote") or []
    adj = adj_items[0].get("adjclose") if adj_items else None
    close = adj if adj and len(adj)==len(ts) else (quote_items[0].get("close") if quote_items else None)
    if not ts or not close:
        raise RuntimeError("Yahoo returned no usable SPY data.")

    spy = pd.DataFrame({
        "d": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        "spy_close": close
    })
    spy["spy_close"] = pd.to_numeric(spy["spy_close"], errors="coerce")
    spy = spy.dropna().drop_duplicates("d", keep="last").sort_values("d")
    spy["spy_ret20"] = 100.0 * (spy["spy_close"]/spy["spy_close"].shift(20)-1.0)
    spy["spy_sma50"] = spy["spy_close"].shift(1).rolling(50,min_periods=50).mean()
    spy["spy_sma200"] = spy["spy_close"].shift(1).rolling(200,min_periods=200).mean()
    spy["spy_above50"] = spy["spy_close"] > spy["spy_sma50"]
    spy["spy_above200"] = spy["spy_close"] > spy["spy_sma200"]
    print(f"[SPY] {len(spy):,} rows loaded", flush=True)
    return spy[["d","spy_ret20","spy_above50","spy_above200"]]

def add_market_context(df, spy):
    z = df.merge(spy, on="d", how="left", sort=False)
    z["rs20_vs_spy"] = z["ret20"] - z["spy_ret20"]
    return z

def train_models(train_df, y_tp, y_sl, y_ret):
    X = train_df[v11.FEATURE_COLS].to_numpy(np.float32)
    tp_model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=220, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.5, random_state=20261005
    )
    sl_model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=220, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.5, random_state=20261006
    )
    ret_model = HistGradientBoostingRegressor(
        learning_rate=.05, max_iter=220, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.5, random_state=20261007
    )
    tp_model.fit(X, y_tp.astype(int))
    sl_model.fit(X, y_sl.astype(int))
    ret_model.fit(X, y_ret.astype(np.float32))
    return tp_model, sl_model, ret_model

def make_scored(df, tp_model, sl_model, ret_model, pw, rw, sl_veto, market_filter):
    X = df[v11.FEATURE_COLS].to_numpy(np.float32)
    p_tp = tp_model.predict_proba(X)[:,1]
    p_sl = sl_model.predict_proba(X)[:,1]
    pred_ret = ret_model.predict(X)

    z = df[["symbol","d","c","ret20","spy_ret20","spy_above50","spy_above200","rs20_vs_spy"]].copy()
    z["p_tp"] = p_tp
    z["p_sl"] = p_sl
    z["pred_return"] = pred_ret

    # V13: SL is a gate, NOT part of the score.
    z = z[z["p_sl"] <= sl_veto].copy()

    if market_filter == "spy_sma50":
        z = z[z["spy_above50"] == True]
    elif market_filter == "spy_sma200":
        z = z[z["spy_above200"] == True]
    elif market_filter == "rs20_positive":
        z = z[z["rs20_vs_spy"] > 0]

    if z.empty:
        return z

    z["tp_rank"] = z.groupby("d")["p_tp"].rank(pct=True, method="average")
    z["ret_rank"] = z.groupby("d")["pred_return"].rank(pct=True, method="average")
    z["score"] = pw*z["tp_rank"] + rw*z["ret_rank"]

    return z.sort_values(["d","score"], ascending=[True,False])

def backtest_with_fallback(scored, base_df, realized, kind, step,
                           max_entries_per_day=1, max_open_positions=10):
    if scored.empty:
        return None, pd.DataFrame()

    pos = pd.Series(np.arange(len(base_df)), index=base_df.index)
    sym_groups = {s:g for s,g in base_df.groupby("symbol", sort=False)}
    rows, last_exit, active = [], {}, []

    for d, day in scored.groupby("d", sort=True):
        d = pd.Timestamp(d)
        # exit_date == current date means slot is free for a new EOD signal.
        active = [x for x in active if x > d]
        slots = max_open_positions - len(active)
        if slots <= 0:
            continue

        want = min(max_entries_per_day, slots)
        taken = 0
        for ix, row in day.sort_values("score", ascending=False).iterrows():
            sym = row["symbol"]
            if sym in last_exit and d <= last_exit[sym]:
                continue

            p = int(pos.loc[ix])
            rr = float(realized[p])
            if not math.isfinite(rr):
                continue

            st = int(step[p]) if math.isfinite(float(step[p])) else 0
            kk = str(kind[p])

            sg = sym_groups[sym]
            loc = np.flatnonzero(sg.index.to_numpy()==ix)
            if len(loc):
                ep = min(len(sg)-1, int(loc[0])+st)
                ed = pd.Timestamp(sg.iloc[ep]["d"])
            else:
                ed = d

            last_exit[sym] = ed
            active.append(ed)
            rows.append({
                "index":int(ix), "symbol":sym, "entry_date":d,
                "entry_price":float(row["c"]),
                "score":float(row["score"]),
                "p_tp":float(row["p_tp"]),
                "p_sl":float(row["p_sl"]),
                "predicted_return":float(row["pred_return"]),
                "exit_kind":kk, "exit_date":ed, "net_return_pct":rr
            })
            taken += 1
            if taken >= want:
                break

    tr = pd.DataFrame(rows)
    if tr.empty:
        return None, tr

    r = tr["net_return_pct"].to_numpy(np.float64)
    wins, losses = r[r>0], r[r<=0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    payoff = float("inf") if not len(losses) else (avg_win/avg_loss if avg_loss>0 else float("inf"))

    m = {
        "n":len(tr),
        "avg_net":float(r.mean()),
        "median_net":float(np.median(r)),
        "win_rate":float(100*(r>0).mean()),
        "avg_win":avg_win,
        "avg_loss_abs":avg_loss,
        "payoff":float(payoff),
        "tp_rate":float(100*(tr["exit_kind"]=="TP").mean()),
        "sl_rate":float(100*(tr["exit_kind"]=="SL").mean()),
        "time_rate":float(100*(tr["exit_kind"]=="TIME").mean())
    }
    return m, tr

def validation_score(m):
    if not m:
        return -1e12
    n = m["n"]
    payoff = m["payoff"] if math.isfinite(m["payoff"]) else 10.0

    # Target hierarchy:
    # 1) 250-350 trades
    # 2) payoff >=4
    # 3) highest avg net
    if 250 <= n <= 350 and payoff >= 4 and m["avg_net"] >= 4:
        tier = 5
    elif 250 <= n <= 350 and payoff >= 4:
        tier = 4
    elif 220 <= n <= 380 and payoff >= 4:
        tier = 3
    elif 250 <= n <= 350:
        tier = 2
    elif 180 <= n <= 420:
        tier = 1
    else:
        tier = 0

    return (
        tier*1_000_000
        + m["avg_net"]*40_000
        + min(payoff,8.0)*4_000
        - abs(n-300)*500
        - m["sl_rate"]*100
    )


def load_checkpoint():
    completed_setups = set()
    rows = []

    if CHECKPOINT_JSON.exists():
        try:
            state = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
            if state.get("version") == VERSION:
                completed_setups = set(int(x) for x in state.get("completed_setups", []))
        except Exception as e:
            print(f"[resume] could not read checkpoint state: {e}", flush=True)

    if CHECKPOINT_CSV.exists():
        try:
            df = pd.read_csv(CHECKPOINT_CSV)
            if "version" in df.columns:
                df = df[df["version"] == VERSION]
            rows = df.to_dict("records")
        except Exception as e:
            print(f"[resume] could not read checkpoint results: {e}", flush=True)

    return completed_setups, rows

def save_checkpoint(completed_setups, rows):
    tmp_csv = CHECKPOINT_CSV.with_suffix(".tmp.csv")
    tmp_json = CHECKPOINT_JSON.with_suffix(".tmp.json")

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(CHECKPOINT_CSV)

    tmp_json.write_text(
        json.dumps(
            {
                "version": VERSION,
                "completed_setups": sorted(int(x) for x in completed_setups),
                "saved_at": pd.Timestamp.now("UTC").isoformat(),
                "rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp_json.replace(CHECKPOINT_JSON)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=.30)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--min-dollar-vol", type=float, default=3_000_000.0)
    ap.add_argument("--reset-checkpoint", action="store_true", help="Clear V13.1 saved progress and start from zero")
    args = ap.parse_args()

    if args.reset_checkpoint:
        for p in (CHECKPOINT_CSV, CHECKPOINT_JSON):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        print("[resume] checkpoint cleared", flush=True)

    print(f"[version] {VERSION}", flush=True)
    print("[plan] V11 score + explicit SL veto + optional SPY/RS gate", flush=True)

    full = v11.build_features(v11.load_daily())
    cand = v11.broad_candidates(full, args.min_price, args.min_dollar_vol)

    spy = yahoo_spy()
    cand_ctx = add_market_context(cand.reset_index().rename(columns={"index":"orig_index"}), spy)
    cand_ctx = cand_ctx.set_index("orig_index")
    # Preserve exact original candidate order/index for outcome alignment.
    cand_ctx = cand_ctx.loc[cand.index]

    completed_setups, all_rows = load_checkpoint()
    if completed_setups:
        print(
            f"[resume] checkpoint found: {len(completed_setups)}/{len(SETUPS)} setups completed",
            flush=True
        )
    else:
        print("[resume] no previous checkpoint; starting fresh", flush=True)

    best = None
    best_payload = None

    for si, (tp, sl, hold) in enumerate(SETUPS, 1):
        if si in completed_setups:
            print(f"[resume] skip completed setup {si}/{len(SETUPS)} TP={tp} SL={sl} HOLD={hold}", flush=True)
            continue

        print(f"[setup {si}/{len(SETUPS)}] TP={tp} SL={sl} HOLD={hold}", flush=True)

        y_tp, ret, step, kind = v11.compute_outcomes(
            full, cand.index.to_numpy(), tp, sl, hold, args.cost
        )
        y_sl = np.array([1.0 if str(x)=="SL" else 0.0 for x in kind], dtype=np.float32)

        usable = np.isfinite(y_tp) & np.isfinite(ret)
        cur = cand_ctx.loc[usable].copy()
        ytp, ysl = y_tp[usable], y_sl[usable]
        rr, ss, kk = ret[usable], step[usable], kind[usable]

        yrs = cur["year"].to_numpy()
        mt, mv, mh = yrs==TRAIN_YEAR, yrs==VALID_YEAR, yrs==HOLDOUT_YEAR

        tp_model, sl_model, ret_model = train_models(
            cur.loc[mt], ytp[mt], ysl[mt], rr[mt]
        )

        Xv = cur.loc[mv, v11.FEATURE_COLS].to_numpy(np.float32)
        try:
            auc_tp = roc_auc_score(ytp[mv], tp_model.predict_proba(Xv)[:,1])
            auc_sl = roc_auc_score(ysl[mv], sl_model.predict_proba(Xv)[:,1])
        except Exception:
            auc_tp, auc_sl = np.nan, np.nan

        val_df = cur.loc[mv]

        for mode, pw, rw in SCORE_MODES:
            for veto in SL_VETOES:
                for mf in MARKET_FILTERS:
                    scored = make_scored(
                        val_df, tp_model, sl_model, ret_model,
                        pw, rw, veto, mf
                    )
                    vm, vtr = backtest_with_fallback(
                        scored, val_df, rr[mv], kk[mv], ss[mv],
                        max_entries_per_day=1, max_open_positions=10
                    )
                    if not vm:
                        continue

                    qualified = (
                        250 <= vm["n"] <= 350 and
                        vm["avg_net"] >= 4.0 and
                        vm["payoff"] >= 4.0
                    )
                    rec = {
                        "version": VERSION,
                        "setup_index": si,
                        "tp":tp, "sl":sl, "hold":hold,
                        "selection_mode":mode,
                        "prob_weight":pw, "return_weight":rw,
                        "sl_veto":veto,
                        "market_filter":mf,
                        "auc_tp_valid":auc_tp,
                        "auc_sl_valid":auc_sl,
                        "valid_n":vm["n"],
                        "valid_avg_net":vm["avg_net"],
                        "valid_win_rate":vm["win_rate"],
                        "valid_payoff":vm["payoff"],
                        "valid_tp_rate":vm["tp_rate"],
                        "valid_sl_rate":vm["sl_rate"],
                        "valid_time_rate":vm["time_rate"],
                        "qualified":int(qualified),
                        "valid_score":validation_score(vm),
                    }
                    all_rows.append(rec)

                    key = (rec["qualified"], rec["valid_score"])
                    if best is None or key > (best["qualified"], best["valid_score"]):
                        best = rec
                        best_payload = (
                            tp_model, sl_model, ret_model,
                            cur, rr, ss, kk, mv, mh, vtr
                        )

        completed_setups.add(si)
        save_checkpoint(completed_setups, all_rows)
        print(
            f"[checkpoint] saved {len(completed_setups)}/{len(SETUPS)} setups",
            flush=True
        )

    if not all_rows:
        raise RuntimeError("No V13.1 result produced.")

    res = pd.DataFrame(all_rows)
    if "version" in res.columns:
        res = res[res["version"] == VERSION]
    res = res.drop_duplicates(
        subset=["setup_index","selection_mode","sl_veto","market_filter"],
        keep="last"
    ).sort_values(
        ["qualified","valid_score"], ascending=[False,False]
    ).reset_index(drop=True)

    if res.empty:
        raise RuntimeError("No V13.1 result produced.")

    res.to_csv(RESULTS, index=False)
    res.head(20).to_csv(TOP20, index=False)

    # Rebuild ONLY the winning setup once. This makes resume safe without
    # serializing sklearn models between sessions.
    best = res.iloc[0].to_dict()
    win_idx = int(best["setup_index"])
    tp, sl, hold = SETUPS[win_idx - 1]
    print(
        f"[final] rebuilding winning setup #{win_idx}: TP={tp} SL={sl} HOLD={hold}",
        flush=True
    )

    y_tp, ret, step, kind = v11.compute_outcomes(
        full, cand.index.to_numpy(), tp, sl, hold, args.cost
    )
    y_sl = np.array([1.0 if str(x)=="SL" else 0.0 for x in kind], dtype=np.float32)

    usable = np.isfinite(y_tp) & np.isfinite(ret)
    cur = cand_ctx.loc[usable].copy()
    ytp, ysl = y_tp[usable], y_sl[usable]
    rr, ss, kk = ret[usable], step[usable], kind[usable]

    yrs = cur["year"].to_numpy()
    mt, mv, mh = yrs==TRAIN_YEAR, yrs==VALID_YEAR, yrs==HOLDOUT_YEAR

    tp_model, sl_model, ret_model = train_models(
        cur.loc[mt], ytp[mt], ysl[mt], rr[mt]
    )

    val_df = cur.loc[mv]
    scored25 = make_scored(
        val_df, tp_model, sl_model, ret_model,
        float(best["prob_weight"]),
        float(best["return_weight"]),
        float(best["sl_veto"]),
        str(best["market_filter"]),
    )
    _, vtr = backtest_with_fallback(
        scored25, val_df, rr[mv], kk[mv], ss[mv],
        max_entries_per_day=1, max_open_positions=10
    )
    vtr.to_csv(TRADES25, index=False)

    # 2026 untouched until after validation winner is selected.
    hold_df = cur.loc[mh]
    scored26 = make_scored(
        hold_df, tp_model, sl_model, ret_model,
        float(best["prob_weight"]),
        float(best["return_weight"]),
        float(best["sl_veto"]),
        str(best["market_filter"]),
    )
    hm, htr = backtest_with_fallback(
        scored26, hold_df, rr[mh], kk[mh], ss[mh],
        max_entries_per_day=1, max_open_positions=10
    )
    htr.to_csv(TRADES26, index=False)

    lines = [
        "US OUTCOME-LEARNING OPTIMIZER V13",
        "="*100,
        f"Version: {VERSION}",
        f"Train={TRAIN_YEAR} | Validate={VALID_YEAR} | Holdout={HOLDOUT_YEAR} partial",
        f"Round-trip cost: {args.cost:.3f}%",
        "",
        "BEST VALIDATION-SELECTED SETUP",
        "-"*100,
    ]
    for k,v in best.items():
        lines.append(f"{k}: {v}")

    lines += ["", "2026 HOLDOUT", "-"*100]
    if hm:
        for k,v in hm.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("No holdout trades.")

    lines += ["", "TOP 20 BY 2025 VALIDATION", "-"*100]
    cols = [
        "qualified","tp","sl","hold","selection_mode","sl_veto","market_filter",
        "auc_tp_valid","auc_sl_valid","valid_n","valid_avg_net","valid_win_rate",
        "valid_payoff","valid_tp_rate","valid_sl_rate","valid_time_rate","valid_score"
    ]
    lines.append(res.head(20)[cols].to_string(index=False))

    lines += ["", "TARGET CHECK", "-"*100]
    v_ok = (
        250 <= best["valid_n"] <= 350 and
        best["valid_avg_net"] >= 4 and best["valid_payoff"] >= 4
    )
    lines.append(f"2025 validation meets full target: {v_ok}")
    if hm:
        h_ok = hm["avg_net"] >= 4 and hm["payoff"] >= 4
        lines.append(f"2026 partial holdout meets return/payoff target: {h_ok}")
        lines.append(f"2026 partial holdout trades so far: {hm['n']}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nSaved:\n{RESULTS}\n{TOP20}\n{REPORT}\n{TRADES25}\n{TRADES26}", flush=True)

if __name__ == "__main__":
    main()
