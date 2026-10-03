RAJIH DAILY RESEARCH — PERSISTENT /data VERSION

All generated CSV/JSON/TXT/checkpoint files now default to:
/data

You can override the folder with:
RAJIH_DATA_DIR=/another/path

IMPORTANT RAILWAY SETUP
1) In the Railway service that runs these scripts, add/mount a Volume.
2) Mount path must be exactly:
   /data
3) Redeploy once.
4) Verify:
   ls -ld /data
   df -h /data
5) Then use the updated scripts.

Recommended run order:
1. python analyze_daily_50pct_tradable30.py
2. python analyze_daily_50pct_deep_research.py
3. python validate_daily_rule_30_diverse_dates.py
4. python compare_daily_validation_success_fail.py
5. python test_daily_rule_all_501_dates.py
6. python count_daily_501_up_down.py
7. python analyze_501_spike_then_fade.py
8. python analyze_validation30_timing.py

Check saved research files:
ls -lah /data

Notes:
- PostgreSQL candle data is already persistent in the database; this change is for generated local research artifacts.
- The daily Yahoo backfill checkpoint now defaults to /data/backfill_us_daily_yahoo_checkpoint.json.
- If /data is not backed by a Railway Volume, files can still disappear on redeploy. The code cannot make an ephemeral filesystem persistent by itself.
