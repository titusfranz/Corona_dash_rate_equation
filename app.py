import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import numpy as np
from scipy.integrate import odeint
import pandas as pd

from get_data import get_corona_data


def f(y, t, params):
    I, S, D, R = y      # unpack current values of y
    rI, rR, dI,ks,ka,q = params  # unpack parameters
    derivs = [rI*I*S*(q*ks+(1-q)*ka)-I*rR-I*dI,      # list of dy/dt=f functions
             -rI*I*S*(q*ks+(1-q)*ka),
             I*dI,
             rR*I]
    return derivs

COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

# Parameters
rI = 0.1          # rate of infection
rR = 0.5          # rate of recovery
dI = 0.01         # rate of death
ks = 0.05         # sociability constant (to symptomatic infected)
ka = 0.1          # sociability constant (to asymptomatic infected)
q = 0.8          # percentage of infected becoming symptomatic

# Initial values
I0 = 1     # initial angular displacement
S0 = 99     # initial angular velocity
D0 = 0.0
R0 = 0.0

# Bundle initial conditions for ODE solver
y0 = [I0, S0, D0, R0]

# Make time array for solution
tStop = 200.
tInc = 0.5
t = np.arange(0., tStop, tInc)

# Call the ODE solver

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(
    children=[

        html.Table(id='corona-data', children=get_corona_data().reset_index().to_json(), style={'display': 'none'}),
        html.Table(id='population-data', style={'display': 'none'}),
        html.H1(children='Corona - Ratenmodell'),

        html.H2(children='Load data'),
        html.Button('Reload data', id='reload-button'),
        html.Button('Fit data', id='fit-button'),
        dcc.Dropdown(id='country-dropdown', value='Germany'),

        dcc.Graph(
            id='simulation-graph',
        ),
        html.Label(id='infection-label'),
        dcc.Slider(
            id='infection-slider',
            min=0,
            max=1,
            step=0.01,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.1,
        ),
        html.Label(id='recovery-label'),
        dcc.Slider(
            id='recovery-slider',
            min=0,
            max=1,
            step=0.01,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.5,
        ),
        html.Label(id='death-label'),
        dcc.Slider(
            id='death-slider',
            min=0,
            max=0.2,
            step=0.001,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.01,
        ),
        html.Label(id='symptomatic-label'),
        dcc.Slider(
            id='symptomatic-slider',
            min=0,
            max=1,
            step=0.01,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.05,
        ),
        html.Label(id='asymptomatic-label'),
        dcc.Slider(
            id='asymptomatic-slider',
            min=0,
            max=1,
            step=0.01,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.1,
        ),
        html.Label(id='percentage-symptomatic-label'),
        dcc.Slider(
            id='percentage-symptomatic-slider',
            min=0,
            max=1,
            step=0.01,
            marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
            value=0.8,
        ),
    ])


@app.callback(
    Output('corona-data', 'children'),
    [Input('reload-button', 'n_clicks')]
)
def load_data(clicks):
    return get_corona_data().reset_index().to_json()


@app.callback(
    Output('population-data', 'children'),
    [Input('reload-button', 'n_clicks')]
)
def load_population_data(clicks):
    population_url = 'https://raw.githubusercontent.com/datasets/population/master/data/population.csv'
    population = pd.read_csv(population_url)
    population = population.set_index(['Country Name', 'Year']).Value
    return population.unstack().idxmax(axis=1).to_json()


@app.callback(
    Output('country-dropdown', 'options'),
    [Input('corona-data', 'children')]
)
def update_country_dropdown(data):
    data = pd.read_json(data)
    countries = data['Country/Region'].unique()
    return [{'label': country, 'value': country} for country in countries]


@app.callback(
    Output('infection-label', 'children'),
    [Input('infection-slider', 'value')]
)
def update_infection(value):
    return 'Infectionsrate: {}'.format(round(value, 2))


@app.callback(
    Output('recovery-label', 'children'),
    [Input('recovery-slider', 'value')]
)
def update_infection(value):
    return 'Genesungsrate: {}'.format(round(value, 2))


@app.callback(
    Output('death-label', 'children'),
    [Input('death-slider', 'value')]
)
def update_infection(value):
    return 'Sterberate: {}'.format(round(value, 2))


@app.callback(
    Output('symptomatic-label', 'children'),
    [Input('symptomatic-slider', 'value')]
)
def update_infection(value):
    return 'Sociability constant (Symptomatisch): {}'.format(round(value, 2))


@app.callback(
    Output('asymptomatic-label', 'children'),
    [Input('asymptomatic-slider', 'value')]
)
def update_infection(value):
    return 'Sociability constant (Asymptomatisch): {}'.format(round(value, 2))


@app.callback(
    Output('percentage-symptomatic-label', 'children'),
    [Input('percentage-symptomatic-slider', 'value')]
)
def update_infection(value):
    return 'Anteil der Infizierten die Symptome zeigen: {}'.format(round(value, 2))


@app.callback(
    Output('simulation-graph', 'figure'),
    [Input('infection-slider', 'value'),
     Input('recovery-slider', 'value'),
     Input('death-slider', 'value'),
     Input('symptomatic-slider', 'value'),
     Input('asymptomatic-slider', 'value'),
     Input('percentage-symptomatic-slider', 'value'),
     Input('corona-data', 'children'),
     Input('population-data', 'children'),
     Input('country-dropdown', 'value')])
def update_figure(infection_rate, recovery_rate, death_rate, symptomatic, asymptomatic, percentage_symptomatic,
                  corona_data, population_data, country):
    data = pd.read_json(corona_data)
    data_country = data.set_index('Country/Region').loc[country]
    data_country = data_country.set_index('Date')
    dates = data_country.index.dayofyear - data_country.index.dayofyear[0]
    population = pd.read_json(population_data, typ='series').loc[country]

    params = [infection_rate, recovery_rate, death_rate, symptomatic, asymptomatic, percentage_symptomatic]
    psoln = odeint(f, y0, dates, args=(params,)) * population

    return {
        'data': [
             {'x': dates, 'y': psoln[:, 0], 'type': 'line', 'name': 'Infizierte'},
             {'x': dates, 'y': psoln[:, 3], 'type': 'line', 'name': 'Genesene'},
             {'x': dates, 'y': psoln[:, 2], 'type': 'line', 'name': 'Verstorbene'},
             # {'x': dates, 'y': data_country['Active Cases'], 'type': 'line', 'name': 'Infizierte'},
             go.Scatter(x=dates, y=data_country['Active Cases'], name='Infizierte',
                        mode='markers', marker_color='#1f77b4')
        ],
        'layout': dict(
            xaxis={'title': 'Zeit'},
            yaxis={'title': 'Prozent'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            transition={'duration': 500},
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
