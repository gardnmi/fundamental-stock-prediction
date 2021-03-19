import altair as alt
from altair.vegalite.v4.schema.channels import Column
import numpy as np
import pandas as pd
from utils import human_format
import streamlit as st


@st.cache(allow_output_mutation=True)
def scatter_filter(data, preset_tickers, filters):
    # Add dataframes
    predictions_df = data['Predictions']
    company_df = data['Company'].set_index('Ticker')
    industry_df = data['Industry']
    share_ratio_df = data['Share Ratio']
    fund_figures_df = data['Fundamental Figures']

    # PRESETS
    if preset_tickers:
        mask = predictions_df.index.get_level_values(0).isin(preset_tickers)
        predictions_df = predictions_df[mask]

    # Reduce to the Latest Prices
    predictions_df = predictions_df[predictions_df.index.get_level_values(
        1) == predictions_df.index.get_level_values(1).max()]

    # FILTERS
    # Sector
    if filters['Sector'] != 'All':
        industry_df = industry_df[industry_df['Sector'].eq(
            filters['Sector'])]

    # Industry
    if filters['Industry'] != 'All':
        industry_df = industry_df[industry_df['Industry'].eq(
            filters['Industry'])]

    # Stock Price
    mask = (predictions_df['Close'] >= filters['Stock Price'][0]) & (
        predictions_df['Close'] <= filters['Stock Price'][1])
    predictions_df = predictions_df[mask]

    # Market Cap
    mask = (share_ratio_df['Market-Cap'] >= filters['Market Cap'][0]
            ) & (share_ratio_df['Market-Cap'] <= filters['Market Cap'][1])
    share_ratio_df = share_ratio_df[mask]

    # Free Cash Flow
    mask = (fund_figures_df['Free Cash Flow'] >= filters['Free Cash Flow'][0]) & (
        fund_figures_df['Free Cash Flow'] <= filters['Free Cash Flow'][1])
    fund_figures_df = fund_figures_df[mask]

    # Total Debt
    mask = (fund_figures_df['Total Debt'] >= filters['Total Debt'][0]) & (
        fund_figures_df['Total Debt'] <= filters['Total Debt'][1])
    fund_figures_df = fund_figures_df[mask]

    # Dividends
    if isinstance(filters['Dividend'], pd.Series):
        mask = predictions_df.index.get_level_values(
            0).isin(filters['Dividend'])
        predictions_df = predictions_df[mask]

    # F-Score
    if filters['F-Score'] != 'All':
        fund_figures_df = fund_figures_df[fund_figures_df['Pietroski F-Score'].eq(
            filters['F-Score'])]

    # Custom Tickers
    if len(filters['Custom Tickers']) > 0:
        mask = predictions_df.index.get_level_values(
            0).isin(filters['Custom Tickers'])
        predictions_df = predictions_df[mask]

    # Merge
    df = predictions_df
    df = df.join(company_df).reset_index()
    df = df.merge(industry_df, on='IndustryId', how='inner')
    df = df.merge(share_ratio_df, on='Ticker', how='inner')
    df = df.merge(fund_figures_df, on='Ticker', how='inner')

    return df


def stock_line_chart(df):

    source = df.copy()
    source = source.rename(columns={'Close': '30 DMA Close'})
    source = source.round(0)
    source = source.reset_index().melt('Date', var_name='category', value_name='Price')
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['Date'], empty='none')

    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='Date:T',
        y='Price:Q',
        color=alt.Color('category:N',
                        legend=alt.Legend(orient='bottom'))
        # color='category:N'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='Date:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Price:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='Date:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    c = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300
    )

    return c


def scatter_variance_chart(df):

    source = df.copy()
    # Calculated Columns
    source['Predicted vs Close %'] = (
        source['Close'] - source['Predicted Close']) / source['Predicted Close']

    bins = np.array([-1, -0.15, 0.15, 999999999999])

    labels = ['< -15%', 'within 15%', '> 15%']

    source['Predicted vs Close % '] = pd.cut(
        source['Predicted vs Close %'], bins=bins, labels=labels, include_lowest=True)

    # Formatting
    # Streamlit cannot handle categorical dtype
    source['Predicted vs Close % '] = source['Predicted vs Close % '].astype(
        str)

    source[['Close', 'Predicted Close']] = source[[
        'Close', 'Predicted Close']].round(0)

    cols = ['Price to Earnings Ratio (ttm)', 'Price to Sales Ratio (ttm)', 'Price to Book Value',
            'Price to Free Cash Flow (ttm)', 'EV/EBITDA', 'EV/Sales', 'EV/FCF', 'Book to Market Value',
            'Operating Income/EV', 'Gross Profit Margin', 'Operating Margin', 'Net Profit Margin',
            'Return on Equity', 'Return on Assets', 'Free Cash Flow to Net Income', 'Current Ratio',
            'Liabilities to Equity Ratio', 'Debt Ratio', 'Earnings Per Share, Basic',
            'Earnings Per Share, Diluted', 'Sales Per Share', 'Equity Per Share', 'Free Cash Flow Per Share',
            'Dividends Per Share']

    source[cols] = source[cols].round(2)

    cols = ['EBITDA', 'Total Debt', 'Free Cash Flow',
            'Market-Cap', 'Enterprise Value']

    for col in cols:
        source[f'{col} ($)'] = source[col].map(human_format)

    source = source.reset_index()
    source = source.rename(columns={'Close': '30 DMA Close'})

    c = alt.Chart(source).mark_point(size=150).encode(
        x='30 DMA Close',
        y='Predicted Close',
        color=alt.Color('Predicted vs Close % ',
                        legend=alt.Legend(orient='bottom')),
        tooltip=[
            'Company Name',
            'Ticker',
            'Sector',
            'Industry',
            '30 DMA Close',
            'Predicted Close',
            alt.Tooltip('Predicted vs Close %:Q', format='.0%'),
            'Market-Cap ($)',
            'Enterprise Value ($)',
            'Free Cash Flow ($)',
            alt.Tooltip('Price to Earnings Ratio (ttm):Q', format='.2f'),
            alt.Tooltip('Price to Sales Ratio (ttm):Q', format='.2f'),
            alt.Tooltip('Price to Book Value:Q', format='.2f'),
            alt.Tooltip('Price to Free Cash Flow (ttm):Q', format='.2f'),
            alt.Tooltip('Current Ratio:Q', format='.2f'),
            alt.Tooltip('Current Ratio:Q', format='.2f'),
            alt.Tooltip('EV/EBITDA:Q', format='.2f'),
            alt.Tooltip('EV/Sales:Q', format='.2f'),
            alt.Tooltip('EV/FCF:Q', format='.2f'),
            'Pietroski F-Score'
        ]
    ).interactive().properties(
        height=500
    )

    return c
