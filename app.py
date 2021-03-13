import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pickle
# from utils import MultiApp
import pathlib

DATA_DIR = pathlib.Path('./data')
MODELS_DIR = pathlib.Path('./models')


@st.cache
def get_data():
    # FEATURES
    common_features_df = pd.read_csv(
        DATA_DIR/'common_features.csv', index_col=['Ticker'])
    banks_features_df = pd.read_csv(
        DATA_DIR/'banks_features.csv', index_col=['Ticker'])
    insurance_features_df = pd.read_csv(
        DATA_DIR/'insurance_features.csv', index_col=['Ticker'])

    # PREDICTIONS
    dfs = []
    for sector in ['common', 'banks', 'insurance']:
        df = pd.read_csv(
            DATA_DIR/f'{sector}_predictions.csv', index_col=['Ticker', 'Date'], parse_dates=['Date'])
        dfs.append(df)

    prediction_df = pd.concat(dfs)

    # SIMILARITY MATRIX
    common_sim_df = pd.read_csv(
        DATA_DIR/'common_sim_matrix.csv', index_col=['Ticker'])
    banks_sim_df = pd.read_csv(
        DATA_DIR/'banks_sim_matrix.csv', index_col=['Ticker'])
    insurance_sim_df = pd.read_csv(
        DATA_DIR/'insurance_sim_matrix.csv', index_col=['Ticker'])

    # TICKERS
    tickers = np.concatenate(
        (common_features_df.index, banks_features_df.index, insurance_features_df.index))

    data_dict = {
        'Predictions': prediction_df,
        'Features': {
            'Common': common_features_df,
            'Banks': banks_features_df,
            'Insurance': insurance_features_df
        },
        'Similarity': {
            'Common': common_sim_df,
            'Banks': banks_sim_df,
            'Insurance': insurance_sim_df
        },
        'Tickers': tickers
    }

    return data_dict


def get_models():

    common_model = pickle.load(open(MODELS_DIR/'common_model.pkl', 'rb'))
    banks_model = pickle.load(open(MODELS_DIR/'banks_model.pkl', 'rb'))
    insurance_model = pickle.load(open(MODELS_DIR/'insurance_model.pkl', 'rb'))

    model_dict = {
        'Common': common_model,
        'Banks': banks_model,
        'Insurance': insurance_model}

    return model_dict


# Loads Data
data = get_data()
models = get_models()

# Header
st.write(""" # Value Stock Finder Put some information **HERE**!""")
ticker = st.selectbox("Choose ticker", data['Tickers'])
st.markdown("<div align='center'><br>"
            "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DATA%20FROM-SIMFIN-blue?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
            "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
st.write('---')

st.line_chart(data['Predictions'].loc[ticker])
st.dataframe(data['Predictions'].loc[ticker])


# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
