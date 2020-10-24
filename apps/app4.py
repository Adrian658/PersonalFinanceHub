import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.dash import no_update

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

import pandas as pd
import os
import numpy as np
import datetime as dt
import requests_cache
import requests

import yfinance as yf
from pandas_datareader import data as pdr

from app import app
from shared.navbar import nav

SESSION = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(days=1))

file_path = os.path.dirname(os.path.abspath(__file__))
outer_dir = "/%s" % os.path.join(*file_path.split('/')[:-2])
ARK_HOLDINGS_PATH = os.path.join(outer_dir, 'Notebooks', 'ARK_Holdings')

FUNDS = []
DAILY_HOLDINGS = {}
TRANSACTION_LOG = {}

for fund in os.listdir(ARK_HOLDINGS_PATH):
    if fund[0] == '.':
        continue
    FUNDS.append(fund)
    df1 = pd.read_csv(os.path.join(ARK_HOLDINGS_PATH, fund, "%s_daily_holdings.csv" % fund))
    df2 = pd.read_csv(os.path.join(ARK_HOLDINGS_PATH, fund, "%s_transaction_log.csv" % fund))
    DAILY_HOLDINGS[fund] = df1
    TRANSACTION_LOG[fund] = df2

# Dropdown to control which service recommendations are shown for
fund_dropdown = html.Div([
    dcc.Dropdown(
        id='fund-dropdown',
        options=[{'label': x, 'value': x} for x in FUNDS],
        value="ARKK",
    )
])

date_picker = html.Div([
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=dt.date(2000, 10, 1),
        max_date_allowed=dt.date(2020, 12, 31),
        initial_visible_month=dt.date(2020, 10, 5),
        end_date=dt.date(2020, 10, 22)
    )
])

today = '2020-10-23'

df = TRANSACTION_LOG['ARKW']
df['date'] = pd.to_datetime(df['date'])
holdings = DAILY_HOLDINGS['ARKW'][DAILY_HOLDINGS['ARKW']['date'] == today].sort_values('weight(%)', ascending=False)

fig = make_subplots(
    rows=4, cols=2,
    column_widths=[0.5, 0.5],
    row_heights=[0.5, 0.5, 0.5, 0.5],
    specs=[[{"colspan": 2}, None],
           [{},             {}],
           [{},             {}],
           [{},             {}]]
)
fig.update_xaxes(tickangle=45)
fig.update_layout(height=1000)

for date, idx in zip(sorted(df['date'].unique(), reverse=True)[:7], [(1,1), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]):
    temp = df[df['date'] == date]
    fig.add_trace(go.Scatter(
        x=temp['ticker'], 
        y=temp['transaction_value'],
        text=temp['shares'],
        mode='markers', name=str(date)),
        row=idx[0], col=idx[1]
    )

new_df = df.groupby(pd.Grouper(key='ticker')).sum().sort_values('transaction_value', ascending=False)
fig1 = go.Figure(layout={'height': 600})
fig1.add_trace(go.Scatter(
                        x=new_df.index, 
                        y=new_df['transaction_value'],
                        text=new_df['shares'],
                        mode='markers', name="All Transactions",
                        marker=dict(size=10)))
fig1.update_xaxes(tickangle=45)


fig2 = go.Figure(layout={'height': 600})
fig2.add_trace(go.Scatter(
                        x=holdings['ticker'], 
                        y=holdings['weight(%)'],
                        text=[(x,y) for x,y in zip(holdings['"market value($)"'], holdings['shares'])],
                        mode='markers', name="All Transactions",
                        marker=dict(size=10)))
fig2.update_xaxes(tickangle=45)

# Main layout of page
layout = html.Div(children=[
    nav,
    fund_dropdown,
    date_picker,
    html.Div(id='test', children="Current Holdings"),
    dcc.Graph(figure=fig2),
    html.Div("Transaction totals over time period"),
    dcc.Graph(id='transactions-graph', figure=fig1),
    html.Div("Daily transactions"),
    dcc.Graph(figure=fig),
    html.Div(id='historical-data-wrap', children=[
        html.Div(children='Historical price graph'),
        dcc.Graph(id='historical-data-graph')
    ], style={'display': 'none'})
])

@app.callback(
    [Output('historical-data-graph', 'figure'), Output('historical-data-wrap', 'style')],
    [Input('transactions-graph', 'clickData')]
)
def test(clickData):
    print(clickData)

    if not clickData:
        return no_update, no_update

    ticker = clickData['points'][0]['x']

    # Query for stock information
    company = yf.Ticker(ticker)

    start_date = "2010-01-01"
    historical_data = pdr.get_data_yahoo(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'), session=SESSION)
    historical_data.index = pd.to_datetime(historical_data.index)

    temp = df[df['ticker'] == ticker].copy(deep=True)

    temp['price'] = historical_data.loc[temp['date']]['Close']
    temp['type'] = ['green' if x else 'red' for x in (temp['shares'] > 0)]
    temp['shares_abs'] = temp['shares'].abs()

    max_size=min(temp['shares_abs'])*10
    temp.at[temp['shares_abs'] > max_size, 'shares_abs'] = max_size
    temp['shares_abs'] = temp['shares_abs'] / (max(temp['shares_abs'])/30)
    temp.at[temp['shares_abs'] < 10, 'shares_abs'] = 10
    print(temp)


    historical_price_fig = go.Figure(layout={'height': 600})
    #historical_price_fig.layout.xaxis.range = (historical_data.index[0], historical_data.index[-1])
    #historical_price_fig = px.scatter(temp, x='date', y='price', 
        #labels={'Shares': 'shares'},
        #color='type',
        #size='shares_abs'
        #color_continuous_scale=[(-5000, "red"), (5000, "blue")]
    #)
    historical_price_fig.add_trace(go.Scatter(
                                            x=historical_data.index, 
                                            y=historical_data['Close'],
                                            mode='lines', name="Price"))
    historical_price_fig.add_trace(go.Scatter(
                                            x=temp['date'], 
                                            y=historical_data.loc[temp['date']]['Close'],
                                            mode='markers', name="Recommendations",
                                            text=temp['shares'],
                                            marker_color=temp['type'],
                                            
                                            marker=dict(size=temp['shares_abs'])))
    
    historical_price_fig.update_xaxes(
        #rangeslider_visible=True,
        #fixedrange=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward")
                #dict(step="all")
            ])
        )        
    )

    #historical_price_fig.update_layout(hovermode='x unified')

    return historical_price_fig, {'display': 'block'}

