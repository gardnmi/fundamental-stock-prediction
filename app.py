from utils import human_format
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pickle
from utils import human_format
from predict import model_explainer
import pathlib
from charts import stock_line_chart, scatter_variance_chart

DATA_DIR = pathlib.Path('./data')
MODELS_DIR = pathlib.Path('./models')

# TODO
# Create Presets for Scatter Plot Filter
#   - Value Stocks
#   - Growth Stocks
#   - Large Cap Stocks
#   - Small Cap Stocks
#   - S&P 500
#   - DOW
#   - NASDAQ
# Format DataFrames https://discuss.streamlit.io/t/cant-adjust-dataframes-decimal-places/1949/4
# Add Filters to Scatter Plot in Sidebar (use the derived csv's)


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

    # COMPANY
    company_df = pd.read_csv(DATA_DIR/'company.csv')
    company_df['Company Name'] = company_df['Company Name'].str.title()

    # INDUSTRY
    industry_df = pd.read_csv(DATA_DIR/'industry.csv')

    # SHARE PRICE RATIOS
    cols = ['Ticker', 'Market-Cap', 'Price to Earnings Ratio (ttm)', 'Price to Sales Ratio (ttm)',
            'Price to Book Value', 'Price to Free Cash Flow (ttm)', 'Enterprise Value', 'EV/EBITDA',
            'EV/Sales', 'EV/FCF', 'Book to Market Value', 'Operating Income/EV']

    stock_derived_df = pd.read_csv(DATA_DIR/'stock_derived.csv')[cols]

    # FUNDAMENTAL RATIOS & FIGURES
    cols = ['Ticker', 'EBITDA', 'Total Debt',
            'Free Cash Flow', 'Gross Profit Margin', 'Operating Margin',
            'Net Profit Margin', 'Return on Equity', 'Return on Assets',
            'Free Cash Flow to Net Income', 'Current Ratio',
            'Liabilities to Equity Ratio', 'Debt Ratio',
            'Earnings Per Share, Basic', 'Earnings Per Share, Diluted',
            'Sales Per Share', 'Equity Per Share', 'Free Cash Flow Per Share',
            'Dividends Per Share', 'Pietroski F-Score']

    dfs = []
    for sector in ['common', 'banks', 'insurance']:
        df = pd.read_csv(
            DATA_DIR/f'{sector}_fundamental_derived.csv')
        dfs.append(df)

    fundamental_derived_df = pd.concat(dfs)[cols]

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
        'Tickers': tickers,
        'Company': company_df,
        'Industry': industry_df,
        'Share Ratio': stock_derived_df,
        'Fundamental Figures': fundamental_derived_df
    }

    return data_dict


@st.cache
def transform_data(data):
    df = data['Predictions']

    # Reduce
    df = df[df.index.get_level_values(1) == df.index.get_level_values(1).max()]

    # Merge
    df.join(data['Company'].set_index(
        'Ticker')).reset_index().merge(data['Industry'], on='IndustryId')
    df = df.merge(data['Share Ratio'], on='Ticker', how='inner')
    df = df.merge(data['Fundamental Figures'], on='Ticker', how='inner')

    # Formatting
    df[['Close', 'Predicted Close']] = df[[
        'Close', 'Predicted Close']].round(0)
    df['Market-Cap (Readable)'] = df['Market-Cap'].map(human_format)

    return df


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
st.title('Random Forrest Stock Valuation')
st.markdown(
    'A machine learning approach to valuing stocks based on Trailing Twelve Month (TTM) Fundamentals')
tickers = data['Company'][['Ticker', 'Company Name']].set_index('Ticker')

st.markdown("<div align='center'><br>"
            "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DATA%20FROM-SIMFIN-blue?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
            "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
st.write('---')


# SCATTER CHART


# df = df.set_index(['Ticker', 'Date'])
st.dataframe(transform_data(data))
# c = scatter_variance_chart(df)
# st.altair_chart(c, use_container_width=True)


ticker = st.selectbox("Choose ticker", tickers.index,
                      format_func=tickers['Company Name'].to_dict().get)

# LINE CHART
df = data['Predictions'].loc[ticker]
c = stock_line_chart(df)
st.altair_chart(c, use_container_width=True)


st.dataframe(df.head())

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
