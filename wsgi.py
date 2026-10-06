from app import create_app

app = create_app()

try:
    from monitor.gann_web import gann_bp
    app.register_blueprint(gann_bp)
    print("[GANN] blueprint registered: /gann", flush=True)
except Exception as e:
    print(f"[GANN] disabled because startup failed: {type(e).__name__}: {e}", flush=True)
