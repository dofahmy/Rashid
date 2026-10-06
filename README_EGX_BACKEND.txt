EGX BACKEND BUNDLE
==================

Files to copy:
- app.py -> /app/app.py
- egx_final_decision_experiment_v2.py -> /app/egx_final_decision_experiment_v2.py
- monitor/egx_live.py -> /app/monitor/egx_live.py
- monitor/egx_worker.py -> /app/monitor/egx_worker.py
- templates/egx_market.html -> /app/templates/egx_market.html
- templates/stocks.html -> /app/templates/stocks.html

Railway:
1) Web service remains unchanged.
2) Add a NEW separate service named EGX Monitor with command:
       python -m monitor.egx_worker
3) Ensure the service shares the same DATABASE_URL.
4) Add build tools for pyswisseph in Dockerfile:
       RUN apt-get update && apt-get install -y gcc g++ make && rm -rf /var/lib/apt/lists/*
5) Add pyswisseph to requirements.txt.

Backend URL:
    /egx-market

The page reads a cached state from the Setting table. The EGX Monitor refreshes
it once after 16:00 Cairo time on Sunday-Thursday.
