from utils import human_format
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import shap
import pickle
import matplotlib.pyplot as plt
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
#   - S&P 500, DOW, NASDAQ (Index etc)
#   - 30,60,90,180,360 MVA
# Format DataFrames https://discuss.streamlit.io/t/cant-adjust-dataframes-decimal-places/1949/4
# Add Filters to Scatter Plot in Sidebar (use the derived csv's)
# Add SHAP Explanations
# Add YFinance Stock Description
# Add Linked Brushing Scatter Plot https://altair-viz.github.io/gallery/scatter_linked_brush.html
# News Feed API?
# Replace SHAP with shapash
# Implement News Feed https://github.com/kotartemiy/pygooglenews

# SideBar
st.set_page_config(
    page_title="Stock Value",
    page_icon="",
    layout='centered',  # wide
    initial_sidebar_state="expanded",
)


@st.cache
def get_data():

    # PREDICTIONS
    dfs = []
    for sector in ['common', 'banks', 'insurance']:
        df = pd.read_csv(
            DATA_DIR/f'{sector}_predictions.csv', index_col=['Ticker', 'Date'], parse_dates=['Date'])
        dfs.append(df)

    prediction_df = pd.concat(dfs)
    _tickers = prediction_df.index.get_level_values(0)

    # FEATURES
    common_features_df = pd.read_csv(
        DATA_DIR/'common_features.csv', index_col=['Ticker']).drop(columns=['Date'])
    banks_features_df = pd.read_csv(
        DATA_DIR/'banks_features.csv', index_col=['Ticker']).drop(columns=['Date'])
    insurance_features_df = pd.read_csv(
        DATA_DIR/'insurance_features.csv', index_col=['Ticker']).drop(columns=['Date'])

    # Remove Tickers not in Predictions

    common_features_df = common_features_df[common_features_df.index.isin(
        _tickers)]

    banks_features_df = banks_features_df[banks_features_df.index.isin(
        _tickers)]

    insurance_features_df = insurance_features_df[insurance_features_df.index.isin(
        _tickers)]

    # SIMILARITY MATRIX
    common_sim_df = pd.read_csv(
        DATA_DIR/'common_sim_matrix.csv', index_col=['Ticker'])
    banks_sim_df = pd.read_csv(
        DATA_DIR/'banks_sim_matrix.csv', index_col=['Ticker'])
    insurance_sim_df = pd.read_csv(
        DATA_DIR/'insurance_sim_matrix.csv', index_col=['Ticker'])

    # COMPANY
    company_df = pd.read_csv(DATA_DIR/'company.csv')
    company_df['Company Name'] = company_df['Company Name'].str.title()
    company_df = company_df[company_df['Ticker'].isin(_tickers)]

    # INDUSTRY
    industry_df = pd.read_csv(DATA_DIR/'industry.csv')

    # TICKERS
    tickers = company_df[['Ticker', 'Company Name']].set_index('Ticker')
    tickers['Company Name (Ticker)'] = tickers['Company Name'] + \
        ' (' + tickers.index + ')'

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

    # Add dataframes
    predictions_df = data['Predictions']
    company_df = data['Company'].set_index('Ticker')
    industry_df = data['Industry']
    share_ratio_df = data['Share Ratio']
    fund_figures_df = data['Fundamental Figures']

    # Reduce
    df = predictions_df[predictions_df.index.get_level_values(
        1) == predictions_df.index.get_level_values(1).max()]

    # Merge
    df = df.join(company_df).reset_index()
    df = df.merge(industry_df, on='IndustryId', how='inner')
    df = df.merge(share_ratio_df, on='Ticker', how='inner')
    df = df.merge(fund_figures_df, on='Ticker', how='inner')

    # Calculated Columns
    df['Predicted vs Close %'] = (
        df['Close'] - df['Predicted Close']) / df['Predicted Close']

    bins = np.array([-1, -0.15, 0.15, 999999999999])

    labels = ['< -15%', 'within 15%', '> 15']

    df['Predicted vs Close % Bin'] = pd.cut(
        df['Predicted vs Close %'], bins=bins, labels=labels, include_lowest=True)

    # Formatting
    # Streamlit cannot handle categorical dtype
    df['Predicted vs Close % Bin'] = df['Predicted vs Close % Bin'].astype(str)

    df[['Close', 'Predicted Close']] = df[[
        'Close', 'Predicted Close']].round(0)

    cols = ['Price to Earnings Ratio (ttm)', 'Price to Sales Ratio (ttm)', 'Price to Book Value',
            'Price to Free Cash Flow (ttm)', 'EV/EBITDA', 'EV/Sales', 'EV/FCF', 'Book to Market Value',
            'Operating Income/EV', 'Gross Profit Margin', 'Operating Margin', 'Net Profit Margin',
            'Return on Equity', 'Return on Assets', 'Free Cash Flow to Net Income', 'Current Ratio',
            'Liabilities to Equity Ratio', 'Debt Ratio', 'Earnings Per Share, Basic',
            'Earnings Per Share, Diluted', 'Sales Per Share', 'Equity Per Share', 'Free Cash Flow Per Share',
            'Dividends Per Share']

    df[cols] = df[cols].round(2)

    cols = ['EBITDA', 'Total Debt', 'Free Cash Flow',
            'Market-Cap', 'Enterprise Value']

    for col in cols:
        df[f'{col} ($)'] = df[col].map(human_format)

    # df['Market-Cap (Readable)'] = df['Market-Cap'].map(human_format)
    # df['Enterprise Value (Readable)'] = df['Enterprise Value'].map(
    #     human_format)
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
df = transform_data(data)

st.dataframe(df.head())


# Variables
COMMON_TICKERS = data['Features']['Common'].index
BANK_TICKERS = data['Features']['Banks'].index
INSURANCE_TICKERS = data['Features']['Insurance'].index
ALL_TICKERS = data['Tickers']

COMMON_MODEL = models['Common']
BANK_MODEL = models['Banks']
INSURANCE_MODEL = models['Insurance']

COMMON_FEATURES = data['Features']['Common']
BANK_FEATURES = data['Features']['Banks']
INSURANCE_FEATURES = data['Features']['Insurance']

COMMON_EXPLAINER = shap.TreeExplainer(COMMON_MODEL)
BANK_EXPLAINER = shap.TreeExplainer(BANK_MODEL)
INSURANCE_EXPLAINER = shap.TreeExplainer(INSURANCE_MODEL)


# Header
with st.beta_container():
    st.title('Random Forrest Stock Valuation')
    st.markdown(
        'A machine learning approach to valuing stocks based on Trailing Twelve Month (TTM) Fundamentals')

    st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/DATA%20FROM-SIMFIN-blue?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
                "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
    st.write('---')

# STOCK FINDER
with st.beta_container():

    c = scatter_variance_chart(df)
    st.altair_chart(c, use_container_width=True)

    with st.beta_expander("See explanation"):
        st.write("""
            TEXT GOES HERE
        """)
        st.image("https://static.streamlit.io/examples/dice.jpg")

    # st.write('---')


# STOCK ANALYSIS
with st.beta_container():

    ticker_dic = ALL_TICKERS['Company Name (Ticker)'].to_dict()
    tickers = st.multiselect("Choose ticker", ALL_TICKERS.index,
                             format_func=ticker_dic.get)

    for ticker in tickers:

        # LINE CHART
        st.markdown(f'> {ticker_dic[ticker]}')
        df = data['Predictions'].loc[ticker]
        c = stock_line_chart(df)
        st.altair_chart(c, use_container_width=True)

        if ticker in COMMON_TICKERS:
            shap_values = COMMON_EXPLAINER(
                COMMON_FEATURES.loc[slice(ticker, ticker), :])[0]
            st.dataframe(COMMON_FEATURES.loc[ticker])
            st.pyplot(shap.waterfall_plot(shap_values))
        elif ticker in BANK_TICKERS:
            pass
        elif ticker in INSURANCE_TICKERS:
            pass
        else:
            pass

# Feature Importance
with st.beta_container():
    st.write('---')
    # https://github.com/slundberg/shap
    # explainer = shap.TreeExplainer(COMMON_MODEL)
    shap_values = COMMON_EXPLAINER(COMMON_FEATURES)
    st.markdown('### Feature Importance')
    shap.plots.bar(shap_values, max_display=15)
    plt.xlabel("Average Absolute Feature Price Movement")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    with st.beta_expander("See explanation"):
        st.write("""
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
        """)
        st.image("https://static.streamlit.io/examples/dice.jpg")
