import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go

import pandas as pd
import os
import re
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

#from app import app
from shared.navbar import nav

### Read in data and format appropriately
bank_stmt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static/bank_statements")

statements = []
for dir_ in os.listdir(bank_stmt_path):
    statements.append(pd.read_csv("%s/%s" % (bank_stmt_path, dir_)))
statement = pd.concat(statements).drop_duplicates().dropna()
statement.set_index('Date', inplace=True)
statement.index = pd.to_datetime(statement.index)

type_dic = {
    "Online Banking Transfer": 'Financial Transfers', 
    'INTERNATIONAL TRANSACTION FEE': 'Internation Transaction Fee', 
    'PURCHASE': 'Discretionary Purchase', 
    'DES': 'Financial Transfers', 
    'DEPOSIT': 'Financial Transfers', 
    'REFUND': 'Refund', 
    'WITHDRWL': 'Withdrawal', 
    'PMNT': 'Financial Transfers'
}

def categorize_payment(transactions):
    for transaction in transactions:
        trans_type = None
        for flag in type_dic.keys():
            match = re.search(flag, transaction)
            if match:
                if trans_type:
                    print("Ambiguous transaction type: ", transaction)
                trans_type = type_dic[match.group()]
        yield trans_type

function=categorize_payment(statement['Description'])
statement['Type'] = list(function)
statement['UnixTime'] = statement.index.astype('int64') // 1e9

### Plot the data
daily_balance = statement[['Running Bal.']].resample('D').mean().apply(lambda x: x.interpolate('pad')).reset_index()

statement_fig = px.line(daily_balance, x="Date", y="Running Bal.", title='Total Account Balance')

transactions_fig = px.scatter(statement.reset_index(), x='Date', y='Amount', color='Type', hover_data=['Description'], marginal_x='histogram', title='Transaction History')

transactions_M_fig = px.bar(statement[statement['Type'] == 'Discretionary Purchase'].resample('M').sum().reset_index(), x='Date', y='Amount', title='Discretionary Spending')

print("Created app 1 data")

layout = html.Div(children=[
    nav,
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.H4(children='Total Account Balance'),
    dcc.Graph(
        id='balance',
        figure=statement_fig
    ),

    html.H4(children='Transactions'),
    dcc.Graph(
        id='scatter',
        figure=transactions_fig
    ),
    html.H4(children='Discretionary Spending'),
    dcc.Graph(
        id='bar',
        figure=transactions_M_fig
    )
])
"""
@app.callback(
    Output('app-1-display-value', 'children'),
    [Input('app-1-dropdown', 'value')]
)
def display_value(value):
    return 'You have selected "{}"'.format(value)
"""