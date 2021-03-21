# from explainerdashboard import ExplainerDashboard

import pandas as pd
import numpy as np
import pickle
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import pathlib
import datetime


DATA_DIR = pathlib.Path('./data')
MODELS_DIR = pathlib.Path('./models')

mtime = pathlib.Path('explainer.joblib').stat().st_mtime
mtime = pd.to_datetime(datetime.datetime.fromtimestamp(mtime))
days_since_update = (pd.to_datetime('today') - mtime).days

if days_since_update > 7:

    model = pickle.load(open(MODELS_DIR/'general_model.pkl', 'rb'))
    y = pd.read_csv(DATA_DIR/'general_target.csv',
                    index_col=['Ticker']).drop(columns=['Date'])
    X = pd.read_csv(DATA_DIR/f'general_features.csv',
                    index_col=['Ticker']).drop(columns=['Date'])

    # Dashboard Explainer is fussy about Column Names
    X.columns = X.columns.str.replace('.', '')
    feature_names = model.get_booster().feature_names
    feature_names = [x.replace('.', '') for x in feature_names]
    model.get_booster().feature_names = feature_names

    explainer = RegressionExplainer(model, X, y)

    db = ExplainerDashboard(
        explainer, title="Stock Valuation Explainer", shap_interaction=False, precision='float32', decision_trees=False)

    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib",
               dump_explainer=True)

db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.flask_server()
