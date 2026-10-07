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
