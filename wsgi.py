from app import create_app
from monitor.gann_web import gann_bp

app = create_app()
app.register_blueprint(gann_bp)
