import pandas as pd
import pickle
import simfin as sf
import shap
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
import pathlib

MODELS_DIR = pathlib.Path('./models')
DATA_DIR = pathlib.Path('./data')


def train(df, winsor_quantile, model_name, feature_name, param):
    df = df.copy()

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
    model = XGBRegressor(**param)
    model.fit(X, y)

    # Save the Model
    pickle.dump(model, open(MODELS_DIR/f"{model_name}.pkl", "wb"))

    # Save Features for SHAP
    X.to_csv(DATA_DIR/f'{feature_name}_features.csv')
    y.to_csv(DATA_DIR/f'{feature_name}_target.csv')

    return model


def predict(model, df, filename):
    df = df.copy()

    X = df.drop(columns=['Close', 'Dataset'])

    df['Predicted Close'] = model.predict(X)

    df[['Close', 'Predicted Close']].to_csv(DATA_DIR/f'{filename}.csv')

    return df


def predict_similiar(model, df, filename, number_of_features=15):
    df = df.copy()

    X = df.drop(columns=['Close', 'Dataset', 'Predicted Close'])

    # Filter Dataset to current Stock Prices Only
    X = X[X.index.get_level_values(1) == X.index.get_level_values(1).max()]

    features = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False).index[:number_of_features]

    tickers = X.index.get_level_values(0)

    similarity_matrix = cosine_similarity(X[features])

    matrix_df = pd.DataFrame(
        similarity_matrix, index=tickers, columns=tickers)

    matrix_df.to_csv(DATA_DIR/f'{filename}.csv')

    return matrix_df
