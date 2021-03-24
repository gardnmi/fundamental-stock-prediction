from utils import human_format, get_default_format
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import yahoo_fin.stock_info as si
from yahoo_fin import news
import shap
import pickle
import matplotlib.pyplot as plt
import pathlib
from charts import stock_line_chart, scatter_variance_chart, scatter_filter
from datetime import datetime
from time import mktime
import reticker
from quotes import quotes
from matplotlib.backends.backend_agg import RendererAgg

DATA_DIR = pathlib.Path('./data')
MODELS_DIR = pathlib.Path('./models')
extractor = reticker.TickerExtractor()


# TODO
# Add Linked Brushing Scatter Plot https://altair-viz.github.io/gallery/scatter_linked_brush.html

# Page Settings
st.set_page_config(
    page_title="Stock Value",
    page_icon="",
    layout='centered',  # wide
    initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)


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
    for sector in ['general', 'banks', 'insurance']:
        df = pd.read_csv(
            DATA_DIR/f'{sector}_predictions.csv', index_col=['Ticker', 'Date'], parse_dates=['Date'])
        dfs.append(df)

    df = pd.concat(dfs)
    _tickers = df.index.get_level_values(0)
    data_dict['Predictions'] = df

    # CURRENT PREDICTIONS
    df = df[df.index.get_level_values(
        1) == df.index.get_level_values(1).max()].reset_index(level=1, drop=True)
    data_dict['Current Predictions'] = df

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

    # ANALYST GROWTH ESTIMATES
    df = pd.read_csv(DATA_DIR/'analyst_growth_estimates.csv')
    df = df.rename(columns={'Unnamed: 0': 'Ticker'}).set_index('Ticker')
    df = df.reindex(data_dict['Current Predictions'].index)
    df = df.merge(data_dict['Company'], left_index=True, right_on='Ticker')
    df = df.merge(data_dict['Industry'],  on='IndustryId')
    df = df[['Next 5 Years (per annum)', 'Industry', 'Ticker']]
    df['Next 5 Years (per annum)'] = df['Next 5 Years (per annum)'].map(lambda x: float(
        x.strip('%').replace(",", ""))/100 if pd.notnull(x) else x)
    df['Growth'] = df.groupby('Industry').transform(
        lambda x: x.fillna(x.mean()))
    df = df[['Ticker', 'Growth']].set_index('Ticker')
    data_dict['Growth Estimates'] = df

    for schema in ['general', 'banks', 'insurance']:
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
        [ratio_figure_dict['General'], ratio_figure_dict['Banks'], ratio_figure_dict['Insurance']])

    return data_dict


def get_models():

    general_model = pickle.load(open(MODELS_DIR/'general_model.pkl', 'rb'))
    banks_model = pickle.load(open(MODELS_DIR/'banks_model.pkl', 'rb'))
    insurance_model = pickle.load(open(MODELS_DIR/'insurance_model.pkl', 'rb'))

    general_explainer = shap.TreeExplainer(general_model)
    banks_explainer = shap.TreeExplainer(banks_model)
    insurance_explainer = shap.TreeExplainer(insurance_model)

    model_dict = {
        'General': general_model,
        'Banks': banks_model,
        'Insurance': insurance_model,
        'General Explainer': general_explainer,
        'Banks Explainer': banks_explainer,
        'Insurance Explainer': insurance_explainer

    }

    return model_dict


# LOAD DATA
data = get_data()
models = get_models()

#### HEADER ####
with st.beta_container():

    st.markdown('''
            <style>
              .typewriter h4 {
                width: auto;
                display: inline-block;
                overflow: hidden;
                border-right: .15em solid white;
                white-space: nowrap;
                margin: 0 auto;
                padding: 0.5em 0px 0.5em;
                letter-spacing: .15em;
                animation: typing 1.5s steps(40, end),
                  blink-caret .75s step-end infinite;
                background: #273746;
                border-radius: 3px;
                color: white;
              }

              @keyframes typing {
                from {
                  max-width: 0;


                }

                to {
                  max-width: 100%;

                }
              }

              @keyframes blink-caret {

                from,
                to {
                  border-color: transparent;
                  box-sizing: border-box;
                  border-right: .8em solid;
                }

                0%,
                49% {
                  border-color: orange;

                }

                50%,
                100% {
                  border-color: #273746;

                }
              }

            </style>
            <div>
              <h1>Stock Valuation w/ Machine Learnimg</h1>
            </div>
            <div class="typewriter">
              <h4> &nbsp; A poor man's bloomberg terminal</h4>
            </div>

                   ''', unsafe_allow_html=True)
    st.markdown(f'>{quotes[np.random.randint(len(quotes)-1)]}')

#### SIDEBAR ####
with st.beta_container():
    #### SUPPORT ####
    with st.sidebar.beta_expander("Support:", expanded=True):
        st.markdown(
            f'''<p><small>You can support by <a href="https://www.buymeacoffee.com/gardnmi" target="_blank">
            buying me a coffee</a>☕️</small></p>''', unsafe_allow_html=True)

    #### ABOUT ####
    with st.sidebar.beta_expander("About:", expanded=True):
        st.markdown(
            f'''<p><small>Traditional valuation models such as DCF are a time consuming process
            that requires a lot of assumptions as inputs. This project aims to simplify and generalize  stock valuation
            using machine learning.  The predicted value, <code>Machine Learning Valuation</code>,
            uses the Trailing Twelve Month Fundamentals as features inputs</small></p>
            <p><small>Visit <a href="https://price-valuation-explainer.herokuapp.com/">Stock Valuation Explainer</a>
            for more detail behind the models predictions.
            </small></p>''', unsafe_allow_html=True)

   #### HOT TO USE ####
    with st.sidebar.beta_expander("How to Use:", expanded=False):
        st.markdown(
            f'''<p><small>Find potential investments in in the Scatter Plot located in the
            <code>Machine Learning Valuation</code> section and input the ticker/company name in the drop down box in the
            <code>Analysis</code> section to further explore. </small></p>
            <p><small>Use the preset filters to filter down the scatter plot
            or customize the filter using the filters below.</small></p>''', unsafe_allow_html=True)

    #### FILTERS ####
    with st.sidebar.beta_expander("Filters:", expanded=False):
        st.markdown(
            '''<p><small> Filters for the <code>Machine Learning Valuation</code> scatter plot.</small></p>''', unsafe_allow_html=True)
        ticker_input = st.text_area(
            'Input Ticker(s)', help='''Tickers can be entered in any format and will
                                       be parsed correctly
                                       \n i.e. "Has GLD/IAU bottomed yet?"
                                       = ['GLD', 'IAU'] or "GLD $IAU" = ['GLD', 'IAU'].
                                       \n Extraction done by https://pypi.org/project/reticker/
                                      ''')

        custom_tickers = extractor.extract(ticker_input)

        options = np.concatenate([['All'], data['Industry'].Sector.unique()])
        sector = st.selectbox('Sector', options=options, help='''A stock market sector is a group of stocks that have a lot in
                                                                 general with each other, usually because they are in similar industries.
                                                                 There are 11 different stock market sectors, according to the most generally
                                                                 used classification system: the Global Industry Classification Standard
                                                                 (GICS)
                                                                 \n https://www.investopedia.com/ask/answers/05/industrysector.asp''')

        options = np.concatenate([['All'], data['Industry'].Industry.unique()])
        industry = st.selectbox('Industry', options=options, help='''Industry refers to a specific group of companies that operate in a similar business sphere.
                                                                     Essentially, industries are created by breaking down sectors into more defined groupings.
                                                                     Therefore, these companies are divided into more specific groups than sectors. Each of the dozen
                                                                     or so sectors will have a varying number of industries, but it can be in the hundreds.
                                                                     \n https://www.investopedia.com/ask/answers/05/industrysector.asp''')

        min_value = data['Current Predictions']['Close'].min()
        max_value = data['Current Predictions']['Close'].max()
        min_stock_price = st.number_input('Min Stock Price',
                                          min_value=min_value,
                                          max_value=max_value,
                                          step=10.0,
                                          value=min_value
                                          )
        max_stock_price = st.number_input('Max Stock Price',
                                          min_value=min_value,
                                          max_value=max_value,
                                          step=10.0,
                                          value=max_value
                                          )

        min_value = int(data['Share Ratio']['Market-Cap'].min())
        max_value = int(data['Share Ratio']['Market-Cap'].max())
        help = '''Market capitalization refers to the total dollar market value of a company's outstanding
                            shares of stock. Generally referred to as "market cap," it is calculated by multiplying the
                            total number of a company's outstanding shares by the current market price of one share.
                            \n https://www.investopedia.com/terms/m/marketcapitalization.asp'''

        min_market_cap = st.number_input('Min Market Cap',
                                         min_value=min_value,
                                         max_value=max_value,
                                         step=100_000,
                                         value=min_value,
                                         format='%d',
                                         help=help
                                         )
        max_market_cap = st.number_input('Max Market Cap',
                                         min_value=min_value,
                                         max_value=max_value,
                                         step=100_000,
                                         value=max_value,
                                         format='%d',
                                         help=help
                                         )

        min_value = int(data['Fundamental Figures']['Free Cash Flow'].min())
        max_value = int(data['Fundamental Figures']['Free Cash Flow'].max())
        help = '''Free cash flow (FCF) represents the cash a company generates after accounting for
                        cash outflows to support operations and maintain its capital assets. Unlike earnings
                        or net income, free cash flow is a measure of profitability that excludes the non-cash
                        expenses of the income statement and includes spending on equipment and assets as well
                        as changes in working capital from the balance sheet.
                        \n https://www.investopedia.com/terms/f/freecashflow.asp'''
        min_fcf = st.number_input('Min Free Cash Flow',
                                  min_value=min_value,
                                  max_value=max_value,
                                  step=100_000,
                                  value=min_value,
                                  format='%d',
                                  help=help
                                  )
        max_fcf = st.number_input('Max Free Cash Flow',
                                  min_value=min_value,
                                  max_value=max_value,
                                  step=100_000,
                                  value=max_value,
                                  format='%d',
                                  help=help
                                  )

        min_value = int(data['Fundamental Figures']['Total Debt'].min())
        max_value = int(data['Fundamental Figures']['Total Debt'].max())
        help = '''Total debt is calculated by adding up a company's liabilities, or debts, which are categorized
                  as short and long-term debt. Financial lenders or business leaders may look at a company's balance
                  sheet to factor in the debt ratio to make informed decisions about future loan options.
                  They calculate the debt ratio by taking the total debt and dividing it by the total assets.
                  \n https://www.indeed.com/career-advice/career-development/how-to-calculate-total-debt#:~:text=Total%20debt%20is%20calculated%20by,decisions%20about%20future%20loan%20options.'''
        min_total_debt = st.number_input('Min Total Debt',
                                         min_value=min_value,
                                         max_value=max_value,
                                         step=100_000,
                                         value=min_value,
                                         format='%d',
                                         help=help
                                         )
        max_total_debt = st.number_input('Max Total Debt',
                                         min_value=min_value,
                                         max_value=max_value,
                                         step=100_000,
                                         value=max_value,
                                         format='%d',
                                         help=help
                                         )

        bool = st.checkbox(
            "Dividend Stocks", value=False, help='''A dividend is the distribution of some of a company's earnings to
                                                     a class of its shareholders, as determined by the company's board of
                                                     directors. General shareholders of dividend-paying companies are typically
                                                     eligible as long as they own the stock before the ex-dividend date.
                                                     Dividends may be paid out as cash or in the form of additional stock.
                                                     \n https://www.investopedia.com/terms/d/dividend.asp''')
        if bool:
            dividend_tickers = data['Fundamental Figures'][data['Fundamental Figures']['Dividends Per Share'].notnull(
            )]['Ticker']
        else:
            dividend_tickers = False

        options = np.concatenate(
            [['All'], data['Fundamental Figures']['Pietroski F-Score'].dropna().sort_values().unique()])
        f_score = st.selectbox('Pietroski F-Score', options=options, help='''Piotroski F-score is a number between 0 and 9
                                                                            which is used to assess strength of company's financial position.
                                                                            The score is used by financial investors in order to find the best
                                                                            value stocks (nine being the best). The score is named after Stanford
                                                                            accounting professor Joseph Piotroski.
                                                                            \n https://en.wikipedia.org/wiki/Piotroski_F-score#:~:text=Piotroski%20F%2Dscore%20is%20a,Stanford%20accounting%20professor%20Joseph%20Piotroski.''')

        filters = {
            'Sector': sector,
            'Industry': industry,
            'Stock Price': [min_stock_price, max_stock_price],
            'Market Cap': [min_market_cap, max_market_cap],
            'Free Cash Flow': [min_fcf, max_fcf],
            'Total Debt': [min_total_debt, max_total_debt],
            'Dividend': dividend_tickers,
            'F-Score': f_score,
            'Custom Tickers': custom_tickers
        }

    #### DISPLAY OPTIONS ####
    with st.sidebar.beta_expander("Display Options:", expanded=False):
        st.markdown(
            f'''<p><small>Uncheck to remove section from <code>Analysis</code>. </small></p>''', unsafe_allow_html=True)

        company_info = st.checkbox(
            'Company Info', value=True)
        analyst_growth = st.checkbox(
            'Analyst Growth Estimates', value=True)
        financial_statements = st.checkbox(
            'Financial Statements', value=True)
        prediction_explanation = st.checkbox(
            'Prediction Explanation', value=True)
        similiar_stocks = st.checkbox(
            'Similar Stocks', value=True)
        news_feed = st.checkbox(
            'News Feed', value=True)
        feature_importance = st.checkbox(
            'Feature Importance', value=True)
        num_sim = st.slider('Number of Similar Stocks',
                            1, 20, value=10, step=1,
                            help='Number of Similar Stocks to Display in the Similar Stock Section')
        num_news = st.slider('Number of News Articles',
                             1, 10, value=5, step=1,
                             help='Number of Articles to Display in the News Feed Section')

    #### CONTACT ####
    with st.sidebar.beta_expander("Contact:", expanded=False):
        st.markdown(
            '- [Github](https://github.com/gardnmi/fundamental-stock-prediction)')

        st.markdown(
            '- [LinkedIn](https://www.linkedin.com/in/michael-gardner-38a29658/)')

    #### CHANGE LOG ####
    with st.sidebar.beta_expander("Change Log:", expanded=False):
        st.markdown(
            '- Initial Commit')


#### Machine Learning Valuation ####
with st.beta_container():
    st.header('** Machine Learning Valuation: **')

    preset_functions = {
        'All': False,
        'S&P500': si.tickers_sp500,
        'DOW': si.tickers_dow,
        'NASDAQ': si.tickers_nasdaq,
        'Day Winners': si.get_day_gainers,
        'Day Losers': si.get_day_losers,
        'Day Most Active': si.get_day_most_active,
        'Undervalued Large Cap': si.get_undervalued_large_caps
    }
    st.markdown('''<p><small> Hover over the circle in the chart
                   to see figures and ratios for each company. Reduce the number of tickers using
                   the preset dropdown below or the filters located in the sidebar.
                </small></p>''', unsafe_allow_html=True)

    presets = st.selectbox('Select a Filter Preset:',
                           options=['All', 'S&P500', 'DOW', 'NASDAQ',
                                    'Day Winners', 'Day Losers', 'Day Most Active',
                                    'Undervalued Large Cap'])

    preset_tickers = preset_functions[presets]

    if preset_tickers:
        try:
            preset_tickers = preset_functions[presets]()

            # Some Functions Return a DataFrame and not a List
            if isinstance(preset_tickers, pd.DataFrame):
                preset_tickers = preset_tickers['Symbol'].to_list()

        except:
            preset_tickers = False
            st.write(
                f"`There is an issue with the {presets} preset.  Please choose another`")

    df = scatter_filter(data, preset_tickers, filters)
    c = scatter_variance_chart(df)
    st.altair_chart(c, use_container_width=True)


#### ANALYSIS ####
with st.beta_container():
    st.header('** Analysis: **')

    # TICKER DROP DOWN
    ticker_dic = data['Tickers']['Company Name (Ticker)'].to_dict()
    tickers = st.multiselect("Choose ticker(s)", data['Tickers'].index.to_list(), default=['ELY'],
                             format_func=ticker_dic.get, help='US Market Only')
    for ticker in tickers:

        if ticker in data['Features']['General'].index:
            key = 'General'
        if ticker in data['Features']['Banks'].index:
            key = 'Banks'
        if ticker in data['Features']['Insurance'].index:
            key = 'Insurance'

        # COMPANY AND STOCK PRICE
        yticker = yf.Ticker(f'{ticker}')
        st.subheader(f'**{ticker_dic[ticker]}**')
        st.markdown(
            f'''<p><small>Current Price: <code>{round(si.get_live_price(ticker),2)}
            </code></small></p>''', unsafe_allow_html=True)

        # COMPANY INFO
        if company_info:
            with st.beta_expander("Company Info", expanded=True):
                st.markdown(f"> {yticker.info['longBusinessSummary']}")

        # CLOSE VS PREDICTED PRICE
        st.subheader(
            '30 Day Moving Avg vs Machine Learning Valuation Over Time')
        df = data['Predictions'].loc[ticker]
        c = stock_line_chart(df)
        st.altair_chart(c, use_container_width=True)

        # GROWTH CALCULATION
        st.markdown('''A considerable limitation in the Machine Learning Valuation 
                       is that it does not equate for growth expectations.  Below you can apply your own
                       growth assumptions for up to 10 years.''')
        growth_estimate_default = data['Growth Estimates'].loc[ticker]['Growth']

        feature_cols = ['Cash, Cash Equivalents & Short Term Investments',
                        'Net Cash from Operating Activities',
                        'Net Income',
                        'Income (Loss) from Continuing Operations',
                        'Earnings Per Share, Basic',
                        'Net Cash from Investing Activities',
                        'Earnings Per Share, Diluted',
                        'Pretax Income (Loss), Adj.',
                        'Gross Profit',
                        'Net Income (Common)']

        input_growth, output_growth = st.beta_columns(2)
        with input_growth:
            help = ''' Enter the number of years of future growth.  
                       5 & 10 years are common in Discounted Cash Flow'''
            years = st.slider('Number of Years', 1, 10,
                              value=5, help=help)
            help = ''' Enter the annualized percent growth expected (i.e. 25%).  
                       Default value uses Yahoo Growth Estimates'''
            growth_input = st.text_input('Avg Growth % (per annum)',
                                         value=f"{growth_estimate_default:.2%}", help=help)

            try:
                growth_input = float(growth_input.strip('%')) / 100
            except:
                st.info('Please enter a percentage i.e. 25%')

        with output_growth:
            try:
                features = data['Features'][key].loc[[ticker]].copy()
                features_growth = features.copy()
                features_growth[feature_cols] = features_growth[feature_cols] * \
                    (1+growth_input)**years

                st.markdown(f'''<h4 align="center"> Machine Learning "Growth" Valuation </h4>
                                <h3 align="center"><code>{models[key].predict(features)[0]:.2f}</code> 
                                &#8594;
                                <code>{models[key].predict(features_growth)[0]:.2f}</code></h3>
                                <br>
                                <p><small font-size=0.75em><cite>Note: The random forest algorithm used cannot extrapolate. 
                                                                 If growth esimates push feature values outside the range seen by the model 
                                                                 then the valuation will just be the highest valuation seen in the data. 
                                                                 <a href ="http://freerangestats.info/blog/2016/12/10/extrapolation" target="_blank">
                                                                 Link for more info. </a>
                                                                 </cite></small></p>
                                ''', unsafe_allow_html=True)
            except:
                st.error(
                    'Oops...Something Went Wrong.')

        # ANALYST ESTIMATES
        if analyst_growth:
            st.subheader('Analyst Growth Estimates')
            st.table(si.get_analysts_info(ticker)[
                'Growth Estimates'].set_index('Growth Estimates')[[ticker]].T)

        # FINANCIAL STATEMENTS
        if financial_statements:
            st.subheader('Financial Statements')
            st.markdown(
                f'''<p><small>
                Publish Date: <code>{data['Income'][key].loc[ticker]['Publish Date']}</code>
                <br>
                Shares (Basic): <code>{data['Income'][key].loc[ticker]['Shares (Basic)']}</code>
                </small></p>
                ''', unsafe_allow_html=True)

            income, balance, cashflow = st.beta_columns(3)
            with income:
                st.write('INCOME')
                st.table(data['Income'][key].loc[ticker].iloc[2:].rename(
                    'TTM').dropna())
            with balance:
                st.write('BALANCE')
                st.table(data['Balance'][key].loc[ticker].iloc[2:].rename(
                    'TTM').dropna())
            with cashflow:
                st.write('CASHFLOW')
                st.table(data['Cashflow'][key].loc[ticker].iloc[2:].rename(
                    'TTM').dropna())

            figures, ratios = st.beta_columns(2)
            with figures:
                st.write('FIGURES')

                figure_df = data['Fundamental Figures'].set_index(
                    'Ticker').loc[ticker].to_frame().T

                # Formatting
                format_dict = get_default_format(figure_df,
                                                 int_format=human_format,
                                                 manual_cols=[
                                                     'EBITDA', 'Total Debt', 'Free Cash Flow'],
                                                 manual_format=human_format)

                for col, value in format_dict.items():
                    figure_df[col] = figure_df[col].apply(
                        value)

                st.table(figure_df.T)
            with ratios:
                st.write('RATIOS')
                ratio_df = data['Share Ratio'].set_index(
                    'Ticker').loc[ticker].to_frame().T

                # Formatting
                format_dict = get_default_format(ratio_df,
                                                 int_format=human_format,
                                                 manual_cols=[
                                                     'Market-Cap', 'Enterprise Value'],
                                                 manual_format=human_format)

                for col, value in format_dict.items():
                    ratio_df[col] = ratio_df[col].apply(
                        value)

                st.table(ratio_df.T)

        # PREDICTION EXPLANATION
        if prediction_explanation:
            st.subheader('Prediction Explanation')
            st.markdown('''<p><small>The waterfall plot is designed to visually display how
                        the values of each feature moves the <code>average</code> stock value to the
                        <code>predicted</code> stock value. Visit
                        <a href="https://price-valuation-explainer.herokuapp.com/" target="_blank">Stock Valuation Explainer</a> for more detail.
                        </small></p>''', unsafe_allow_html=True)

            shap_values = models[f'{key} Explainer'](
                data['Features'][key].loc[slice(ticker, ticker), :])[0]

            _lock = RendererAgg.lock
            with _lock:
                shap.waterfall_plot(shap_values, max_display=20)
                st.pyplot()

        # SIMILIAR STOCKS
        if similiar_stocks:
            st.subheader('Similiar Stocks')
            st.markdown(
                '''<p><small>The 10 most similiar stocks using <code>cosine similarity</small></p>''',
                unsafe_allow_html=True)

            cols = ['Ticker', 'Close', 'Predicted Close', 'Company Name', 'Sector', 'Industry', 'Market-Cap', 'Enterprise Value',
                    'Price to Earnings Ratio (ttm)', 'Price to Sales Ratio (ttm)', 'Price to Book Value',
                    'Price to Free Cash Flow (ttm)', 'EV/EBITDA', 'EV/Sales', 'EV/FCF', 'Book to Market Value',
                    'Operating Income/EV', 'EBITDA', 'Total Debt', 'Free Cash Flow', 'Gross Profit Margin',
                    'Operating Margin', 'Net Profit Margin', 'Return on Equity',
                    'Return on Assets', 'Free Cash Flow to Net Income', 'Current Ratio', 'Liabilities to Equity Ratio',
                    'Debt Ratio', 'Earnings Per Share, Basic', 'Earnings Per Share, Diluted', 'Sales Per Share',
                    'Equity Per Share', 'Free Cash Flow Per Share', 'Dividends Per Share', 'Pietroski F-Score']

            df = data['Similarity'][key].loc[ticker].sort_values(
                ascending=False).iloc[1:num_sim+1].to_frame()
            df.index.name = 'Ticker'
            df = df.join(data['Current Predictions'])
            df = df.join(data['Company'].set_index('Ticker')).reset_index()
            df = df.merge(data['Industry'], on='IndustryId', how='inner')
            df = df.merge(data['Share Ratio'], on='Ticker', how='inner')
            df = df.merge(data['Fundamental Figures'],
                          on='Ticker', how='inner')
            df = df[cols]
            df = df.set_index('Ticker')
            manual_cols = ['Market-Cap', 'Enterprise Value',
                           'EBITDA', 'Total Debt', 'Free Cash Flow']

            format_dict = get_default_format(df,
                                             int_format=human_format,
                                             manual_cols=manual_cols,
                                             manual_format=human_format)
            for key, value in format_dict.items():
                df[key] = df[key].apply(
                    value)

            df = df.rename(columns={'Close': '30 DMA',
                                    'Predicted Close': 'ML Valuation'})
            st.dataframe(df)

        # NEWS FEED
        if news_feed:
            st.subheader('News Feed')
            for article in news.get_yf_rss(ticker)[:num_news+1]:
                published_parsed = news.get_yf_rss(
                    'ELY')[0]['published_parsed']
                published = datetime.fromtimestamp(
                    mktime(published_parsed)).strftime("%m/%d/%Y")
                st.markdown(
                    f'''<h4><a href="{article['link']}" target="_blank">{article['title']} </a>
                    <span style="font-size:.80rem"> - Published: {published} </span></h4>
                        ''', unsafe_allow_html=True)
                st.markdown(
                    f'''<p> <small> {article['summary']} </small> </p> ''', unsafe_allow_html=True)

# Feature Importance
if feature_importance:
    with st.beta_container():
        # https://github.com/slundberg/shap
        shap_values = shap.TreeExplainer(
            models['General'])(data['Features']['General'])
        st.subheader('Feature Importance')
        st.markdown(
            ''' <p> <small> The relative importance of each <code> fundamental feature </code >
            in the predicted stock price </small> </p>''', unsafe_allow_html=True)

        _lock = RendererAgg.lock
        with _lock:
            shap.plots.bar(shap_values, max_display=15)
            plt.xlabel("Average Absolute Feature Price Movement")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()


st.markdown("<div align='center'><br>"
            "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DATA%20FROM-SIMFIN%20AND%20YAHOO_FIN-blue?style=for-the-badge'"
            "alt='API stability' height='25'/>"
            "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
            "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
st.write('---')
