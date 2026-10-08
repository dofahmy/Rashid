SEVEN REPEAT ACCUMULATION STUDY - EXPERIMENTAL, READ-ONLY
========================================================
Command (Railway shell, existing backend Python environment):
  python -m monitor.seven_accumulation_study --market egypt --output-dir /tmp/seven-study-egypt
  python -m monitor.seven_accumulation_study --market sp500 --output-dir /tmp/seven-study-sp500
Quick smoke run:
  python -m monitor.seven_accumulation_study --market egypt --max-symbols 30 --output-dir /tmp/seven-smoke

Output:
  egypt_historical_events.csv: every historical repeat and matched controls; full forward outcomes
  egypt_cohort_summary.csv: hit-rate by time horizon for each cohort
  egypt_study_info.json: methodology and warnings

Cohorts:
- repeat: all historical Double-7 Repeat clusters from original engine
- random_matched: one non-repeat date in same stock and calendar quarter for every repeat
- consolidation_no_repeat: random-matched dates whose trailing 20 days contracted (range ratio <= .8),
  volume in second half <=1.2x first half and absolute 20-session price change <=15%.
  This is a research-only proxy, not proof of institutional accumulation.

Forward calculations use NEXT SESSION OPEN and subsequent daily H/L, not Repeat Price.
Same-session stop & target = AMBIGUOUS. Incomplete follow-up => no future hit-rate denominator.
Missing open/high/low => no result for the affected measurement (no guessed values).
EGX OHLC columns depend on local DB schema; no reliable adjusted-OHLC if only raw OHLC
and adjusted close exist. Verify local corporate-action adjustments before trusting outputs.
SP500 OHLC is rescaled by adj_close/raw_close per session to match adjusted close.

LIMITATIONS:
This is an exploratory matched-control test. It is NOT an out-of-sample predictive validation.
Quarter matching controls calendar timing partially but not sector, price, liquidity,
volatility, market regime or selection/survivorship bias. Repeat price-clustering can be
non-causal. Expand controls + walk-forward test before making investment decisions.
Not connected to app routes, so current dashboard behavior is unchanged.
