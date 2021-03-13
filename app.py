import pandas as pd
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
    common_pred_df = pd.read_csv(
        DATA_DIR/'banks_predictions.csv', index_col=['Ticker', 'Date'])
    banks_pred_df = pd.read_csv(
        DATA_DIR/'banks_predictions.csv', index_col=['Ticker', 'Date'])
    insurance_pred_df = pd.read_csv(
        DATA_DIR/'banks_predictions.csv', index_col=['Ticker', 'Date'])

    # SIMILARITY MATRIX
    common_sim_df = pd.read_csv(
        DATA_DIR/'common_sim_matrix.csv', index_col=['Ticker'])
    banks_sim_df = pd.read_csv(
        DATA_DIR/'banks_sim_matrix.csv', index_col=['Ticker'])
    insurance_sim_df = pd.read_csv(
        DATA_DIR/'insurance_sim_matrix.csv', index_col=['Ticker'])

    data_dict = {
        'Common': {
            'Features': common_features_df,
            'Predictions': common_pred_df,
            'Matrix': common_sim_df
        },
        'Banks': {
            'Features': banks_features_df,
            'Predictions': banks_pred_df,
            'Matrix': banks_sim_df
        },
        'Insurance': {
            'Features': insurance_features_df,
            'Predictions': insurance_pred_df,
            'Matrix': insurance_sim_df
        }
    }

    return data_dict


@st.cache
def get_models():

    common_model = pickle.load(open(MODELS_DIR/'common_model.pkl', 'rb'))
    banks_model = pickle.load(open(MODELS_DIR/'banks_model.pkl', 'rb'))
    insurance_model = pickle.load(open(MODELS_DIR/'insurance_model.pkl', 'rb'))

    model_dict = {
        'Common': common_model,
        'Banks': banks_model,
        'Insurance': insurance_model}

    return model_dict


# Header
st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')


# Loads the Boston House Price Dataset
data = get_data()
models = get_models()


stock = st.selectbox(
    "Choose countries", list(df.index), ["China", "United States of America"]
)

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
