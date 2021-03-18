import altair as alt
import numpy as np
import pandas as pd
from utils import human_format


def stock_line_chart(df):

    source = df.copy()
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


def scatter_variance_chart(data):

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

    df = df.reset_index()
    c = alt.Chart(df).mark_point(size=60).encode(
        x='Close',
        y='Predicted Close',
        color=alt.Color('Predicted vs Close % Bin',
                        legend=alt.Legend(orient='bottom')),
        tooltip=[
            'Company Name',
            'Ticker',
            'Sector',
            'Industry',
            'Close',
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
