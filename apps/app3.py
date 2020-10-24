import warnings

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash import no_update

import bs4

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go

import pandas as pd
import numpy as np
import datetime as dt
import requests_cache
import requests
import sqlite3

from pymongo import MongoClient

import yfinance as yf
from pandas_datareader import data as pdr

from app import app
from shared.navbar import nav
from shared.navbar import generate_table
import time


### GLOBAL CONSTANTS ###

MF_RECS_COLLECTIONS_MAP = {
    "Stock Advisor: Buy Recommendations": "sa_buy_recs", 
    "Stock Advisor: Best Buys Now": "sa_bbn_recs",
    "Rule Breakers: Buy Recommendations": "rb_buy_recs",
    "Rule Breakers: Best Buys Now": "rb_bbn_recs",
    'All': 'all'
}

SESSION = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=dt.timedelta(days=1))


### GLOBAL LAYOUT VARIABLES ###

# Dropdown to control which service recommendations are shown for
service_dropdown = html.Div([
    dcc.Dropdown(
        id='service-dropdown',
        options=[{'label': x, 'value': y} for x, y in MF_RECS_COLLECTIONS_MAP.items()],
        value="sa_buy_recs",
    )
])

# Slider to control recommendations shown by count of occurences
recommendation_count_slider = html.Div(children=[
    dcc.RangeSlider(
        id='recommendation-count-slider',
        min=1,
        max=50,
        step=1,
        value=[0, 50],
    ),
    html.Div(id='recommendation-count-slider-values')
])

# Main layout of page
layout = html.Div(children=[
    # Navbar
    nav,
    # Page title information
    html.H1(id='Main', children=['Hello Dash']),
    html.Div(children='''Dash: A web application framework for Python.'''),
    # Recommendation graph and corresponding controls
    service_dropdown,
    recommendation_count_slider,
    dcc.Graph(id='recommendations-graph'),
    # Selected recommendation display 
    html.Div(id='recommendation-container', children=[
        html.H3(id='recommendation-name'),
        dcc.Graph(id='recommendation-historical-graph'),
        html.H5(id='recommendation-promotion'),
        html.P(id='recommendation-date'),
        html.Div(id='recommendation-url'),
        html.P(id='recommendation-description', style={'whitespace': 'pre-line'})
    ], style={'display': 'none'}),
    # Storage components
    dcc.Store(id='recommendations-df'),
    dcc.Store(id='recommendation-ticker-storage'),
    dcc.Store(id='recommendation-date-storage'),
    dcc.Store(id='date-boundaries')
])

### FUNCTIONS USED INTERNALLY ###

# Query MongoDB for MF recommendation information
def mongo_query(database, collection=None, params={}, host='localhost'):
    myclient = MongoClient(host)
    db = myclient.get_database(database)

    if collection == 'all':
        data_coll = []
        for collection_iter in MF_RECS_COLLECTIONS_MAP.values():
            if collection_iter == 'all':
                continue
            collection_ref = db[collection_iter]
            temp_data = collection_ref.find(params)
            data_coll.append(pd.DataFrame(temp_data))
        data = pd.concat(data_coll)
    else:
        collection_ref = db[collection]
        data = collection_ref.find(params)
    myclient.close()
    return data

# Filter dataframe by counts to find number of times each stock has been recommended
def filter_counts(df, max_occur, min_occur):
    value_counts = df['ticker'].value_counts()
    return df.set_index('ticker')[value_counts.between(min_occur, max_occur)].reset_index()

# Returns all recommendations and selected recommendation from data stored in browser
def get_recommendations(ticker, rec_date, data):

    df = pd.DataFrame(data)
    df['recommend_date'] = pd.to_datetime(df['recommend_date'])
    recommendation_all = df[df['ticker'] == ticker]
    recommendation_chosen = recommendation_all[recommendation_all['recommend_date'] == rec_date]
    if len(recommendation_chosen) != 1:
        return 404
    recommendation_chosen = recommendation_chosen.iloc[0]

    return recommendation_all, recommendation_chosen

# Returns the historical data graph and company specific metadata
def create_historical_data_graph(ticker, recommendation_df):

    # Query for stock information
    company = yf.Ticker(ticker)

    start_date = "2010-01-01"
    historical_data = pdr.get_data_yahoo(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'), session=SESSION)
    historical_data.index = pd.to_datetime(historical_data.index)
    recommendation_df = recommendation_df[recommendation_df['recommend_date'] >= start_date]

    historical_price_fig = go.Figure(layout={'height': 600})
    historical_price_fig.layout.xaxis.range = (historical_data.index[0], historical_data.index[-1])
    historical_price_fig.add_trace(go.Scatter(
                                            x=historical_data.index, 
                                            y=historical_data['Close'],
                                            mode='lines', name="Price"))
    historical_price_fig.add_trace(go.Scatter(
                                            x=recommendation_df['recommend_date'], 
                                            y=historical_data.loc[recommendation_df['recommend_date']]['Close'],
                                            mode='markers', name="Recommendations",
                                            marker=dict(size=10)))
    
    historical_price_fig.update_xaxes(
        rangeslider_visible=False,
        fixedrange=True,
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

    historical_price_fig.update_layout(hovermode='x unified')
    

    

    return historical_price_fig, company

# Finds the triggering element's id
def find_trigger_id(attr=0):
    #print(dash.callback_context.triggered)
    return dash.callback_context.triggered[0]['prop_id'].split('.')[attr]

# Convert html to list of dash components
def convert_html_to_dash(element):
    if type(element) == bs4.element.NavigableString:
        return str(element)
    else:
        name = element.name
        contents = [convert_html_to_dash(x) for x in element.contents if type(element) != bs4.element.Comment]

        try:
            return getattr(html,name.title())(contents)
        except:
            return contents


### DASH CALLBACKS FUNCTIONS ###

# TRIGGER: Service-dropdown is updated to change service type selection
# UPDATES: Recommendations (dataframe) stored in browser
#          Max and current values of recommendation-count-slider
@app.callback(
    Output('recommendations-df', 'data'),
    [Input('service-dropdown', 'value')]
)
def update_recommendations_data(collection_name):

    # Retrieve data from MDB and filter to exclude recommendations before 2015
    collection_data = mongo_query('investment-db', collection_name)
    df = pd.DataFrame(collection_data).drop('_id', axis=1)
    df = df[df['recommend_date'] > '2015-01-01']
    df.sort_values('recommend_date', ascending=False, inplace=True)

    return df.to_dict('records')

### TRIGGER: Resizing the view range of recommendations-graph
### OUTPUTS: Update the values of count-slider

@app.callback(
    [Output('recommendation-count-slider', 'max'), Output('recommendation-count-slider', 'value')],
    [Input('recommendations-df', 'data')],#Input('recommendations-graph', 'relayoutData'), 
    [State('recommendation-count-slider', 'value')]
)
def test(data, slider_value):

    if not data:
        print("No data")
        return no_update, no_update

    df = pd.DataFrame(data)
    df['recommend_date'] = pd.to_datetime(df['recommend_date'])

    #if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
    #    df = df[df['recommend_date'].between(relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]'])]

    start, end = df.iloc[-1]['recommend_date'], df.iloc[0]['recommend_date']
    print("Setting: ", start, end)

    # Store counts df so as to not recompute when slider is updated?
    counts = df['ticker'].value_counts()
    max_c = max(counts)
    print(max_c)

    min_selected = slider_value[0]

    if min_selected >= max_c:
        min_selected = 1
    
    return max_c, [1, max_c]

# TRIGGER: Recommendation dataframe (in browser) is changed or recommendation-count-slider is updated
# UPDATES: Recommendations displayed on main recommendations-graph
#          Range slider display values
@app.callback(
    [Output('recommendations-graph', 'figure'), Output('recommendation-count-slider-values', 'children')],
    [Input('recommendation-count-slider', 'value'), Input('recommendations-df', 'data')],
    [State('recommendations-graph', 'figure')]
)
def update_recommendations_graph(slider_value, data, recommendations_graph):
    if not data:
        return no_update, no_update

    # Transform current json data to df
    df = pd.DataFrame(data)
    df['recommend_date'] = pd.to_datetime(df['recommend_date'])
        
    # If the slider value is changed don't update the slider, update the figure and the display values
    if find_trigger_id() == 'recommendation-count-slider':

        # Store min and max of slider selection
        max_selected = slider_value[1]
        min_selected = slider_value[0]

        if recommendations_graph:
            start, end = recommendations_graph['layout']['xaxis']['range'][0], recommendations_graph['layout']['xaxis']['range'][1]
            df = df[df['recommend_date'].between(start, end)]

        df = filter_counts(df, max_selected, min_selected)
        print(df.head(), df.tail())

        # Create recommendations figure
        recommendations_fig = px.scatter(df,
            x='recommend_date', 
            y='ticker', 
            color='ticker',
            title="Motley Fool Recommendations",
            labels={
                'promotion': 'Headline: ',
                'recommend_date': 'Recommended ',
                'url': 'URL: '
            },
            height=800
        )

        #start, end = recommendations_fig.layout.xaxis.range[0], recommendations_fig.layout.xaxis.range[-1]

        # When the recommendation slider is updated, and the callback is not creating the initial figure
        if find_trigger_id() == 'recommendation-count-slider' and recommendations_graph and recommendations_graph['layout']['xaxis']['range']:
            #recommendations_fig.layout = recommendations_graph['layout']
            recommendations_fig.layout.xaxis.range = recommendations_graph['layout']['xaxis']['range']
            #recommendations_fig.layout.yaxis.range = recommendations_graph['layout']['yaxis']['range']

        return recommendations_fig, 'Number of recommendations  -  Min: %s, Max: %s' % (min_selected, max_selected)
    
    elif find_trigger_id() == 'recommendations-df':

         # Find the counts of tickers
        counts = df['ticker'].value_counts()
        max_c = max(counts)
        print("3: ", max_c)
        
        # Create recommendations figure
        recommendations_fig = px.scatter(df,
            x='recommend_date', 
            y='ticker', 
            color='ticker',
            title="Motley Fool Recommendations",
            labels={
                'promotion': 'Headline: ',
                'recommend_date': 'Recommended ',
                'url': 'URL: '
            },
            height=800
        )

        return recommendations_fig, 'Number of recommendations  -  Min: %s, Max: %s' % (1, max_c)

# TRIGGER: Main recommendation or historical data graph is clicked on
# UPDATES: Recommendation ticker (in storage) and date (in storage)
@app.callback(
    [Output('recommendation-ticker-storage', 'data'), Output('recommendation-date-storage', 'data')],
    [Input('recommendations-graph', 'clickData'), Input('recommendation-historical-graph', 'clickData')]
)
def update_selected_recommendation(selected_g1, selected_g2):

    # Do not proceed if callback is made during bootup
    if not selected_g1 and not selected_g2:
        raise PreventUpdate

    # Extract info from event
    trigger_id = find_trigger_id()

    # Lambda to extract selection variables
    extract_selection = lambda x, idx: (x['points'][idx]['y'], x['points'][idx]['x'])

    # Triggering element is recommendations-graph
    # Update both the ticker and recommendation date
    if trigger_id == 'recommendations-graph':
        ticker, rec_date = extract_selection(selected_g1, 0)
        return_tuple = (ticker, rec_date)
    # Triggering element is historical-price-graph
    # Only update the recommendation date
    else:
        ticker, rec_date = extract_selection(selected_g2, 1)
        return_tuple = (no_update, rec_date)

    return return_tuple

# TRIGGER: Recommendation date (in storage) or ticker (in storage) is updated
# UPDATES: Recommendation data depending on the activated input
@app.callback(
    [Output('recommendation-name', 'children'), Output('recommendation-historical-graph', 'figure'),
     Output('recommendation-promotion', 'children'), Output('recommendation-date', 'children'), 
     Output('recommendation-url', 'children'), Output('recommendation-description', 'children'),
     Output('recommendation-container', 'style')],
    [Input('recommendation-ticker-storage', 'data'), Input('recommendation-date-storage', 'data')],
    [State('recommendations-df', 'data')]
)
def edit_recommendation(ticker, rec_date, data):

    # Do not proceed if callback is made during bootup
    if not ticker:
        #print("Preventing Update", rec_date)
        raise PreventUpdate
    
    # Find trigger id
    trigger_id = find_trigger_id()

    # Find the matching ticker in dataframe
    recommendations_df, recommendation = get_recommendations(ticker, rec_date, data)

    # Extract information from dataframe
    promotion, description, url = recommendation['promotion'], \
                                    recommendation['description'], \
                                    recommendation['url']

    # Convert the html description to dash components
    description = convert_html_to_dash(bs4.BeautifulSoup(description, features="lxml").contents[0].contents[0])

    # Set default no update for recommendation independent variables
    recommendation_graph, company_name = no_update, no_update 

    # Only update the recommendation independent variables if the selected stock changes
    # If we select a new recommendation for the same stock, we don't need to recreate the graph or requery for stock metadata
    if trigger_id == 'recommendation-ticker-storage':
        recommendation_graph, company = create_historical_data_graph(ticker, recommendations_df)
        company_name = company.info['longName']

    return company_name, \
        recommendation_graph, \
        promotion, \
        rec_date, \
        html.Div(children=[html.Span('Click '), html.A('here ', href=url), html.Span(' to view original article')]), \
        description, \
        {'display': 'block'}