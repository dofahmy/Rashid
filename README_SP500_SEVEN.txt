S&P 500 — نظام الـ 7
===================

New pages:
    /sp500-seven
    /sp500-seven/screener
    /sp500-seven/day/YYYY-MM-DD

Button:
    SCREEN SP500

What it does
------------
Same Double-7 system used for Egypt:
- whole S&P 500 constituent universe
- individual stock
- S&P 500 index (^GSPC)
- price signal, volume signal, strict Double 7
- same-stock Repeat Price + Time
- SCREEN SP500 for latest qualifying Repeat Price+Time clusters
- market-day drilldown showing individual stocks with 7 / Double-7 signals

Constituent list
----------------
The refresh job downloads the CURRENT S&P 500 constituent table from Wikipedia.
The index often contains slightly more than 500 ticker rows because some
companies have multiple listed share classes. We intentionally keep every
current constituent row rather than arbitrarily dropping share classes.

Database
--------
Creates dedicated tables automatically:
    sp500_seven_constituents
    sp500_seven_daily
    sp500_seven_index

First run
---------
Open /sp500-seven and click:
    Update S&P500 Data

The first load is intentionally much heavier than later updates because it
backfills history. Default:
    SP500_HISTORY_START=2000-01-01

You can shorten it with a Railway variable, e.g.:
    SP500_HISTORY_START=2010-01-01

Later updates only refresh the recent rolling window.

Requirements
------------
yfinance>=0.2.54
lxml>=5.0
html5lib>=1.1
requests
pandas
numpy

No extra Railway service is required. The Update button starts a detached
one-shot refresh job from the Web service.
