#!/usr/bin/env python3
import math
from pathlib import Path
import numpy as np
import us_setup_optimizer_v10_resume as v10

OUT = Path('/data')
OUT.mkdir(parents=True, exist_ok=True)

SETUPS = [
    {'name': 'A', 'tp': 20.0, 'sl': 4.0, 'hold': 10, 'selection_mode': 'top1'},
    {'name': 'B', 'tp': 24.0, 'sl': 4.0, 'hold': 10, 'selection_mode': 'top1'},
]
COST = 0.30


def run_setup(full, cand, setup):
    y, r, step, kind = v10.compute_outcomes(
        full, cand.index.to_numpy(), setup['tp'], setup['sl'], setup['hold'], COST
    )

    usable = np.isfinite(y) & np.isfinite(r)
    cur = cand.loc[usable].copy()
    yy = y[usable]
    rr = r[usable]
    ss = step[usable]
    kk = kind[usable]

    years = cur['year'].to_numpy()
    train = years == 2024
    test = years == 2026

    print(f"[{setup['name']}] 2024 train={train.sum():,} | 2026 test={test.sum():,}", flush=True)

    model = v10.train_model(cur.loc[train], yy[train])
    test_df = cur.loc[test]
    prob = model.predict_proba(test_df[v10.FEATURE_COLS].to_numpy(np.float32))[:, 1]
    selected = v10.choose_daily_ranked(test_df, prob, setup['selection_mode'])
    metrics, trades = v10.backtest_selected(
        selected, test_df, rr[test], kk[test], ss[test]
    )
    return metrics, trades


def main():
    print('Loading V10 data and features...', flush=True)
    full = v10.build_features(v10.load_daily())
    cand = v10.broad_candidates(full, 2.0, 3_000_000.0)

    report = []
    for setup in SETUPS:
        print(
            f"\nTesting Setup {setup['name']} TP={setup['tp']} SL={setup['sl']} Hold={setup['hold']}",
            flush=True,
        )
        m, trades = run_setup(full, cand, setup)
        trades.to_csv(OUT / f"us_2026_setup_{setup['name']}_trades.csv", index=False)

        report += [
            '', '=' * 70, f"SETUP {setup['name']}", '=' * 70,
            f"TP +{setup['tp']}% | SL -{setup['sl']}% | Hold {setup['hold']} | top1",
            f"Trades: {m['n']}",
            f"Average net: {m['avg_net']:.4f}%",
            f"Median net: {m['median_net']:.4f}%",
            f"Win rate: {m['win_rate']:.2f}%",
            f"Average winner: {m['avg_win']:.4f}%",
            f"Average loser: -{m['avg_loss_abs']:.4f}%",
            f"Payoff: {m['payoff']:.4f}:1" if math.isfinite(m['payoff']) else 'Payoff: infinite',
            f"TP exits: {m['tp_rate']:.2f}%",
            f"SL exits: {m['sl_rate']:.2f}%",
            f"Time exits: {m['time_rate']:.2f}%",
        ]

    text = '\n'.join(report)
    print(text)
    (OUT / 'us_2026_two_setups_report.txt').write_text(text, encoding='utf-8')

    print('\nSaved:')
    print('/data/us_2026_two_setups_report.txt')
    print('/data/us_2026_setup_A_trades.csv')
    print('/data/us_2026_setup_B_trades.csv')


if __name__ == '__main__':
    main()
