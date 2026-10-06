#!/usr/bin/env python3
import traceback

print("1) importing app/create_app...")
from app import create_app
print("OK")

print("2) creating Flask app...")
app = create_app()
print("OK")

print("3) importing Gann blueprint...")
try:
    from monitor.gann_web import gann_bp
    print("OK")
except Exception:
    traceback.print_exc()
    raise

print("4) registering blueprint...")
app.register_blueprint(gann_bp)
print("OK")

print("5) routes containing gann:")
for r in app.url_map.iter_rules():
    if "gann" in r.rule.lower():
        print(r.rule, "->", r.endpoint)

print("GANN STARTUP CHECK PASSED")
