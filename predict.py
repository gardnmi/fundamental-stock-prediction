import pandas as pd
import numpy as np
import pickle
import simfin as sf
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
from load import load_dataset
import pathlib
import os
from dotenv import load_dotenv

load_dotenv()
SIMFIN_API_KEY = os.getenv('SIMFIN_API_KEY', 'free')
MODELS_DIR = pathlib.Path('./models')
DATA_DIR = pathlib.Path('./data')


def train(df, winsor_quantile, model_name, feature_name, model_input):

    # Filter Dataset to current Stock Prices Only
    model_df = df[df.index.get_level_values(
        1) == df.index.get_level_values(1).max()]

    # Winsorize the data to even out the distribution
    model_df = sf.winsorize(model_df, clip=True, columns=[
                            'Close'], quantile=winsor_quantile)

    # DataFrames with signals for training- and test-sets.
    X = model_df.drop(columns=['Close', 'Dataset'])
    y = model_df['Close']

    # Fit Model
    model = XGBRegressor(kwargs=model_input)
    model.fit(X, y)

    # Save the Model
    pickle.dump(model, open(MODELS_DIR/f"{model_name}.pkl", "wb"))

    # Save Features for SHAP
    X.to_csv(DATA_DIR/f'{feature_name}_features.csv')

    return model


def predict(model, df, filename):

    X = df.drop(columns=['Close', 'Dataset'])

    df['Predicted Close'] = model.predict(X)

    df[['Close', 'Predicted Close']].to_csv(DATA_DIR/f'{filename}.csv')

    return df


def predict_similiar(model, df, filename, number_of_features=15):
    X = df.drop(columns=['Close', 'Dataset', 'Predicted Close'])

    # Filter Dataset to current Stock Prices Only
    X = X[X.index.get_level_values(1) == X.index.get_level_values(1).max()]

    features = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False).index[:number_of_features]

    tickers = X.index.get_level_values(0)

    similarity_matrix = cosine_similarity(X[features])

    matrix_df = pd.DataFrame(similarity_matrix, index=tickers, columns=tickers)

    matrix_df.to_csv(DATA_DIR/f'{filename}.csv')

    return matrix_df


def model_explainer(model, df):
    X = df.drop(columns=['Close', 'Dataset', 'Predicted Close'])

    # Filter Dataset to current Stock Prices Only
    X = X[X.index.get_level_values(1) == X.index.get_level_values(1).max()]

    explainer = shap.TreeExplainer(model)

    explainer(X)


# LOAD
common_df = load_dataset(dataset='common', simfin_api_key=SIMFIN_API_KEY)
banks_df = load_dataset(dataset='banks', simfin_api_key=SIMFIN_API_KEY)
insurance_df = load_dataset(dataset='insurance', simfin_api_key=SIMFIN_API_KEY)

company_df = sf.load_companies(market='us', refresh_days=1)
industry_df = sf.load_industries(refresh_days=1)

company_df.to_csv(DATA_DIR/'company.csv')
company_df.to_csv(DATA_DIR/'industry.csv')

# TRAIN
common_model = train(common_df,
                     winsor_quantile=0.01,
                     model_name='common_model',
                     feature_name='common',
                     model_input=dict(learning_rate=0.01,
                                      max_depth=2,
                                      subsample=.5,
                                      colsample_bylevel=0.7,
                                      colsample_bytree=0.7,
                                      n_estimators=210))

banks_model = train(banks_df,
                    winsor_quantile=0.05,
                    model_name='banks_model',
                    feature_name='banks',
                    model_input=dict(learning_rate=0.01,
                                     max_depth=2,
                                     subsample=.8,
                                     colsample_bylevel=0.7,
                                     colsample_bytree=0.7,
                                     n_estimators=200))

insurance_model = train(insurance_df,
                        winsor_quantile=0.08,
                        model_name='insurance_model',
                        feature_name='insurance',
                        model_input=dict(learning_rate=0.01,
                                         max_depth=2,
                                         subsample=1,
                                         colsample_bylevel=0.7,
                                         colsample_bytree=0.7,
                                         n_estimators=150))

# PREDICT
common_df = predict(common_model, common_df, 'common_predictions')
banks_df = predict(banks_model, banks_df, 'banks_predictions')
insurance_df = predict(insurance_model, insurance_df, 'insurance_predictions')

# PREDICT SIMILIAR STOCKS
common_matrix_df = predict_similiar(
    common_model, common_df, 'common_sim_matrix')
banks_matrix_df = predict_similiar(banks_model, banks_df, 'banks_sim_matrix')
insurance_matrix_df = predict_similiar(
    insurance_model, insurance_df, 'insurance_sim_matrix')
