import pandas as pd
import numpy as np
import pickle
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import pathlib

DATA_DIR = pathlib.Path('./data')
MODELS_DIR = pathlib.Path('./models')


model = pickle.load(open(MODELS_DIR/'common_model.pkl', 'rb'))
y = pd.read_csv(DATA_DIR/'common_target.csv',
                index_col=['Ticker']).drop(columns=['Date'])
X = pd.read_csv(DATA_DIR/f'common_features.csv',
                index_col=['Ticker']).drop(columns=['Date'])

# Dashboard Explainer is fussy about Column Names
X.columns = X.columns.str.replace('.', '')
feature_names = model.get_booster().feature_names
feature_names = [x.replace('.', '') for x in feature_names]
model.get_booster().feature_names = feature_names


explainer = RegressionExplainer(model, X, y)

db = ExplainerDashboard(explainer, title="Stock Valuation Explainer")
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib",
           dump_explainer=True)
server = db.flask_server()
