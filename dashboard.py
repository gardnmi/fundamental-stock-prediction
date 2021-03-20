from explainerdashboard import ExplainerDashboard

db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.flask_server()
