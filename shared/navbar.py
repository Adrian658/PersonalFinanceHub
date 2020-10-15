import dash_bootstrap_components as dbc
import dash_html_components as html


nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("App 1", href="/apps/app1")),
        dbc.NavItem(dbc.NavLink("App 2", href="/apps/app2")),
        dbc.NavItem(dbc.NavLink("App 3", href="/apps/app3")),
        dbc.NavItem(dbc.NavLink("Active", active=True, href="#")),
        dbc.NavItem(dbc.NavLink("A link", href="#")),
        dbc.NavItem(dbc.NavLink("Disabled", disabled=True, href="#")),
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem("Item 1"), dbc.DropdownMenuItem("Item 2")],
            label="Dropdown",
            nav=True,
        ),
    ]
)

def generate_table(dataframe, max_rows=1000):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])