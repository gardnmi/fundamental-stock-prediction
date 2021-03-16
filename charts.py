import altair as alt
import streamlit as st


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
        color='category:N'
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
    # source = source[source.index.get_level_values(
    #     1) == source.index.get_level_values(1).max()]
    source = source.reset_index()
    c = alt.Chart(source).mark_point(size=60).encode(
        x='Close',
        y='Predicted Close',
        color=alt.Color('Predicted vs Close % Bin',
                        legend=alt.Legend(orient='bottom')),
        tooltip=['Sector',
                 'Industry',
                 'Company Name',
                 'Ticker',
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
