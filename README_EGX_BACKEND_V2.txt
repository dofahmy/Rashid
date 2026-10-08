EGX BACKEND BUNDLE V2 — NOW + DATE + REFRESH

Replace/copy:
app.py -> /app/app.py
egx_final_decision_experiment_v2.py -> /app/egx_final_decision_experiment_v2.py
monitor/egx_live.py -> /app/monitor/egx_live.py
monitor/egx_worker.py -> /app/monitor/egx_worker.py
monitor/egx_data_refresh.py -> /app/monitor/egx_data_refresh.py
monitor/egx_refresh_once.py -> /app/monitor/egx_refresh_once.py
templates/egx_market.html -> /app/templates/egx_market.html
templates/stocks.html -> /app/templates/stocks.html

New:
- NOW or historical date selection.
- Historical AS-OF analysis uses only data on/before selected date.
- Refresh button updates EGX daily data and recalculates NOW state.
- Neutral/Conflict shows separate BUY and SELL tables.
- Automatic EGX worker retries only every 5 minutes after failures.

Railway EGX Monitor start command:
python -m monitor.egx_worker

Variables:
DATABASE_URL=<same PostgreSQL>
EGX_REFRESH_HOUR=16
EGX_RESEARCH_SCRIPT=/app/egx_final_decision_experiment_v2.py

requirements.txt additions:
pyswisseph>=2.10
yfinance>=0.2.54

Dockerfile if pyswisseph needs compilation:
RUN apt-get update && apt-get install -y gcc g++ make && rm -rf /var/lib/apt/lists/*


V2.1 FIX
--------
- Historical date selection now uses the actual EGX stock calendar, not the
  synthetic index proxy. This fixes false "No market data exists..." errors.
- NOW mode automatically rebuilds and saves state if the cache is missing after
  a deploy/version-key change.


V2.2 TABLE SORTING
------------------
- Added Net Buy Score = Buy score - Sell score.
- Buy score, Sell score, and Net Buy Score headers are clickable.
- Each click toggles descending/ascending numeric sorting.


V3.3 — DOUBLE 7 CHART + MARKET DRILLDOWN
----------------------------------------
- Stock table now includes the actual stock close on each Double-7 date.
- Stock chart plots actual close with Double-7 markers.
- Marker rule: higher than previous Double-7 level -> up arrow below price;
  lower -> down arrow above price; first/equal -> diamond.
- Market mode adds synthetic index value on every row and charts that index.
- Market mode counts how many individual stocks also made strict Double 7 on
  the same date.
- Clicking the date/count opens /egx-seven/day/YYYY-MM-DD with the stock list.


V3.6 — MARKET REPEAT PRICE+TIME SCREENER
----------------------------------------
New page:
    /egx-seven/screener

Screens ALL EGX stocks and returns the latest qualifying repeated same-stock
Double-7 price/time cluster for each stock.

Controls:
- price tolerance %
- maximum session gap
- recency window (30/60/90/180/365 days or all history)

Columns include:
- symbol
- latest repeat date
- first date in cluster
- repeat price
- touches
- max session gap
- price spread %
- current close
- current vs repeat price %
- age in days
- all cluster dates/prices
