import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import os
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import robin_stocks as rs
from pymongo import MongoClient

from app import app
from shared.navbar import nav
from shared.navbar import generate_table

"""
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

#robin_positions_all['ticker'] = robin_positions_all['instrument'].apply(lambda x: rs.stocks.get_symbol_by_url(x))

robin_positions_all = robin_positions_all[['ticker', 'quantity', 'average_buy_price', 'created_at', 'updated_at']]
robin_positions_all['quantity'] = robin_positions_all['quantity'].astype('float64').round(2)
robin_positions_all['average_buy_price'] = robin_positions_all['average_buy_price'].astype('float64').round(2)
robin_positions_all['created_at'] = pd.DatetimeIndex(robin_positions_all['created_at']).tz_convert('US/Central').tz_localize(None).round('1s')
robin_positions_all['updated_at'] = pd.DatetimeIndex(robin_positions_all['updated_at']).tz_convert('US/Central').tz_localize(None).round('1s')
"""
print('Created app 2 data')

layout = html.Div(children=[
    nav,
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for Python.
    ''')
    #generate_table(robin_positions_all)
])