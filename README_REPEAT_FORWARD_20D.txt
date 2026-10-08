ADDED: forward 20-session Repeat evaluation for SCREEN EGYPT and SCREEN SP500.
Preserved original Repeat clusters, Gap, Direction, returns, columns and routes.
New files / changes:
 monitor/seven_forward_20d.py (new)
 monitor/seven_system.py
 monitor/sp500_seven_system.py
 templates/seven_screener.html
 templates/sp500_seven_screener.html

Definitions:
- Baseline: Repeat Price (cluster average, consistent with original display).
- Begin observing on the trading session AFTER the latest Repeat date.
- Max Rise: max next-20 daily HIGH relative to Repeat Price.
- Max Drawdown: min next-20 daily LOW relative to Repeat Price.
- +5% before -3% and +5% before -5% are calculated separately.
- TARGET_FIRST / STOP_FIRST / AMBIGUOUS (both levels within same session)
  / NEITHER (20 sessions complete, no crossing) / PENDING / NO_DATA.
- First-hit session column reports the session of first crossing (or ambiguous touch).
- 20-session coverage shows observed count; pending does NOT mean failure.
- EGX needs available high/low or adjusted high/low in the existing daily table.
  If absent, forward result shows NO_DATA (no fabricated highs/lows).
- Same-bar ordering cannot be established from daily OHLC. Intraday data needed.
- This is descriptive price-path analysis, not verified executable P&L.
- Existing direction logic on SP500 remains unchanged.
