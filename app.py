from utils import human_format, get_default_format
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import shap
import pickle
import matplotlib.pyplot as plt
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


# Page Settings
st.set_page_config(
    page_title="Stock Value",
    page_icon="",
    layout='centered',  # wide
    initial_sidebar_state="expanded",
)


@st.cache
def get_data():
    data_dict = {}
    feature_dict = {}
    similarity_dict = {}
    income_dict = {}
    balance_dict = {}
    cashflow_dict = {}
    ratio_figure_dict = {}

    # PREDICTIONS
    dfs = []
    for sector in ['common', 'banks', 'insurance']:
        df = pd.read_csv(
            DATA_DIR/f'{sector}_predictions.csv', index_col=['Ticker', 'Date'], parse_dates=['Date'])
        dfs.append(df)

    df = pd.concat(dfs)
    _tickers = df.index.get_level_values(0)
    data_dict['Predictions'] = df

    # COMPANY & TICKERS
    df = pd.read_csv(DATA_DIR/'company.csv')
    df['Company Name'] = df['Company Name'].str.title()
    df = df[df['Ticker'].isin(_tickers)]
    data_dict['Company'] = df

    df = df[['Ticker', 'Company Name']].set_index('Ticker')
    df['Company Name (Ticker)'] = df['Company Name'] + \
        ' (' + df.index + ')'
    data_dict['Tickers'] = df

    # INDUSTRY
    df = pd.read_csv(DATA_DIR/'industry.csv')
    data_dict['Industry'] = df

    # SHARE PRICE RATIOS
    cols = ['Ticker', 'Market-Cap', 'Price to Earnings Ratio (ttm)', 'Price to Sales Ratio (ttm)',
            'Price to Book Value', 'Price to Free Cash Flow (ttm)', 'Enterprise Value', 'EV/EBITDA',
            'EV/Sales', 'EV/FCF', 'Book to Market Value', 'Operating Income/EV']

    df = pd.read_csv(DATA_DIR/'stock_derived.csv')[cols]
    data_dict['Share Ratio'] = df

    for schema in ['common', 'banks', 'insurance']:
        # FEATURES
        df = pd.read_csv(
            DATA_DIR/f'{schema}_features.csv', index_col=['Ticker']).drop(columns=['Date'])
        df = df[df.index.isin(_tickers)]

        feature_dict[schema.title()] = df

        # SIMILARITY MATRIX
        df = pd.read_csv(
            DATA_DIR/f'{schema}_sim_matrix.csv', index_col=['Ticker'])
        similarity_dict[schema.title()] = df

        # INCOME STATEMENT
        df = pd.read_csv(
            DATA_DIR/f'{schema}_income.csv', index_col=['Ticker'])

        # format dataframe
        format_dict = get_default_format(
            df, int_format=human_format, float_format=human_format)
        for key, value in format_dict.items():
            df[key] = df[key].apply(value)

        drop_cols = ['SimFinId', 'Currency', 'Fiscal Year',
                     'Fiscal Period',  'Restated Date',  'Shares (Diluted)']
        df = df.drop(columns=drop_cols)

        income_dict[schema.title()] = df

        # BALANCE SHEET
        df = pd.read_csv(
            DATA_DIR/f'{schema}_balance.csv', index_col=['Ticker'])

        format_dict = get_default_format(
            df, int_format=human_format, float_format=human_format)
        for key, value in format_dict.items():
            df[key] = df[key].apply(value)

        drop_cols = ['SimFinId', 'Currency', 'Fiscal Year',
                     'Fiscal Period',  'Restated Date',  'Shares (Diluted)']
        df = df.drop(columns=drop_cols)

        balance_dict[schema.title()] = df

        # CASH FLOW STATEMENT
        df = pd.read_csv(
            DATA_DIR/f'{schema}_cashflow.csv', index_col=['Ticker'])

        format_dict = get_default_format(
            df, int_format=human_format, float_format=human_format)
        for key, value in format_dict.items():
            df[key] = df[key].apply(
                value)

        drop_cols = ['SimFinId', 'Currency', 'Fiscal Year',
                     'Fiscal Period',  'Restated Date',  'Shares (Diluted)']
        df = df.drop(columns=drop_cols)
        cashflow_dict[schema.title()] = df

        # FUNDAMENTAL RATIOS & FIGURES
        df = pd.read_csv(DATA_DIR/f'{schema}_fundamental_derived.csv')

        cols = ['Ticker', 'EBITDA', 'Total Debt',
                'Free Cash Flow', 'Gross Profit Margin', 'Operating Margin',
                'Net Profit Margin', 'Return on Equity', 'Return on Assets',
                'Free Cash Flow to Net Income', 'Current Ratio',
                'Liabilities to Equity Ratio', 'Debt Ratio',
                'Earnings Per Share, Basic', 'Earnings Per Share, Diluted',
                'Sales Per Share', 'Equity Per Share', 'Free Cash Flow Per Share',
                'Dividends Per Share', 'Pietroski F-Score']
        df = df[cols]
        ratio_figure_dict[schema.title()] = df

    data_dict['Features'] = feature_dict
    data_dict['Similarity'] = similarity_dict
    data_dict['Income'] = income_dict
    data_dict['Balance'] = balance_dict
    data_dict['Cashflow'] = cashflow_dict
    data_dict['Fundamental Figures'] = pd.concat(
        [ratio_figure_dict['Common'], ratio_figure_dict['Banks'], ratio_figure_dict['Insurance']])

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

COMMON_SIM = data['Similarity']['Common']
BANKS_SIM = data['Similarity']['Banks']
INSURANCE_SIM = data['Similarity']['Insurance']

COMMON_INCOME = data['Income']['Common']
BANK_INCOME = data['Income']['Banks']
INSURANCE_INCOME = data['Income']['Insurance']

COMMON_BALANCE = data['Balance']['Common']
BANK_BALANCE = data['Balance']['Banks']
INSURANCE_BALANCE = data['Balance']['Insurance']

COMMON_CASHFLOW = data['Cashflow']['Common']
BANK_CASHFLOW = data['Cashflow']['Banks']
INSURANCE_CASHFLOW = data['Cashflow']['Insurance']

# Header
with st.beta_container():
    st.title('Random Forrest Stock Valuation')
    st.markdown(
        'A machine learning approach to valuing stocks based on Trailing Twelve Month (TTM) Fundamentals')

    # st.markdown("<div align='center'><br>"
    #             "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
    #             "alt='API stability' height='25'/>"
    #             "<img src='https://img.shields.io/badge/DATA%20FROM-SIMFIN-blue?style=for-the-badge'"
    #             "alt='API stability' height='25'/>"
    #             "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
    #             "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
    # st.write('---')

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
        yticker = yf.Ticker(f'{ticker}')
        st.header(f'**{ticker_dic[ticker]}**')

        with st.beta_expander("Company Info"):
            st.markdown(f"> {yticker.info['longBusinessSummary']}")
        st.markdown(' ', unsafe_allow_html=True)

        # LINE CHART
        st.subheader('Close vs Valuation')
        df = data['Predictions'].loc[ticker]
        c = stock_line_chart(df)
        st.altair_chart(c, use_container_width=True)

        if ticker in COMMON_TICKERS:
            st.subheader('Financials')
            st.write(
                f"Publish Date: `{COMMON_INCOME.loc[ticker]['Publish Date']}`")
            st.write(
                f"Shares (Basic): `{COMMON_INCOME.loc[ticker]['Shares (Basic)']}`")

            income, balance, cashflow = st.beta_columns(3)
            with income:
                st.write('INCOME')
                st.table(COMMON_INCOME.loc[ticker].iloc[2:].rename(
                    'TTM').dropna())
            with balance:
                st.write('BALANCE')
                st.table(COMMON_BALANCE.loc[ticker].iloc[2:].rename(
                    'TTM').dropna())
            with cashflow:
                st.write('CASHFLOW')
                st.table(COMMON_CASHFLOW.loc[ticker].iloc[2:].rename(
                    'TTM').dropna())

            shap_values = COMMON_EXPLAINER(
                COMMON_FEATURES.loc[slice(ticker, ticker), :])[0]
            st.pyplot(shap.waterfall_plot(shap_values))

            # Make an Input to Expand Beyond 5
            st.markdown('### 5 Most Similiar Stocks')
            st.dataframe(COMMON_SIM.loc[ticker].sort_values(
                ascending=False).iloc[1:6])

        elif ticker in BANK_TICKERS:
            pass
        elif ticker in INSURANCE_TICKERS:
            pass
        else:
            pass

        st.write('---')

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
