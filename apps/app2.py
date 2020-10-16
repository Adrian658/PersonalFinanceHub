import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go

import pandas as pd
import os
import numpy as np
import requests_cache
import time

import robin_stocks as rs
from pymongo import MongoClient

from app import app
from shared.navbar import nav
from shared.navbar import generate_table

sesh = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=60*10)
#sesh.headers = rs.globals.SESSION.headers
rs.globals.SESSION = sesh


start = time.time()
robin_user = os.environ.get('robinhood_username')
robin_pswd = os.environ.get('robinhood_pswd')

robin_login = rs.login(robin_user, robin_pswd, by_sms=True)

robin_positions_all = pd.DataFrame(rs.account.get_all_positions())

# Connet to MongoDB to get ticker info
myclient = MongoClient("localhost")
db = myclient.get_database('investment-db')
robin_instruments_coll = db['robin_instruments']
robin_instr_to_ticker = {}
for instr in robin_instruments_coll.find():
    robin_instr_to_ticker[instr['instrument']] = instr['ticker']

robin_positions_all['ticker'] = robin_positions_all['instrument'].apply(lambda x: robin_instr_to_ticker[x.split('/')[-2]])

robin_positions_all = robin_positions_all[['ticker', 'quantity', 'average_buy_price', 'created_at', 'updated_at']]
robin_positions_all['quantity'] = robin_positions_all['quantity'].astype('float64').round(2)
robin_positions_all['average_buy_price'] = robin_positions_all['average_buy_price'].astype('float64').round(2)
robin_positions_all['created_at'] = pd.DatetimeIndex(robin_positions_all['created_at']).tz_convert('US/Central').tz_localize(None).round('1s')
robin_positions_all['updated_at'] = pd.DatetimeIndex(robin_positions_all['updated_at']).tz_convert('US/Central').tz_localize(None).round('1s')

robin_positions_current = robin_positions_all[robin_positions_all['quantity'] > 0].copy(deep=True)
quotes = rs.stocks.get_quotes(list(robin_positions_current['ticker']))
robin_positions_current.loc[:, 'current_price'] = [x['last_extended_hours_trade_price'] for x in quotes]
robin_positions_current.loc[:, 'current_price'] = robin_positions_current['current_price'].astype('float64')
robin_positions_current.loc[:, 'equity'] = robin_positions_current['quantity'] * robin_positions_current['current_price']
print(time.time() - start)

portfolio_summary = go.Figure(layout={'height': 600, 'width': 600})
portfolio_summary.add_trace(go.Pie(
    labels=robin_positions_current['ticker'],
    values=robin_positions_current['equity'],
    textinfo="label+percent",
    text=robin_positions_current['equity'],
    hovertemplate=
    '<b>Ticker</b>: %{label}'+
    '<br><b>Position</b>: %{percent}</br>'+
    '<b>Equity</b>: %{text:$.2f}'+
    '<extra></extra>'
))
portfolio_summary.update_layout(
    title_text="Portfolio summary"#,
    #margin=dict(t=0, b=0, l=0, r=0)
)

print('Created app 2 data')

layout = html.Div(children=[
    nav,
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    dcc.Graph(figure=portfolio_summary)
    #generate_table(robin_positions_all)
])