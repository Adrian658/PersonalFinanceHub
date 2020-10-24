import warnings
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import plotly
    import plotly.express as px
    import plotly.graph_objs as go

from app import app
#from apps import app1, app3, app2, app4
from apps import app4

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.validation_layout = html.Div([
    app.layout,
    #app1.layout,
    #app2.layout,
    #app3.layout,
    app4.layout
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return app.layout
    #elif pathname == '/apps/app1':
    #    return app1.layout
    #elif pathname == '/apps/app2':
    #    return app2.layout
    #elif pathname == '/apps/app3':
    #    return app3.layout
    elif pathname == '/apps/app4':
        return app4.layout
    else:
        return '404' + str(pathname)

if __name__ == '__main__':
    app.run_server(debug=True)