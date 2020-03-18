import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

import numpy as np
from scipy.integrate import odeint


def f(y, t, params):
    I, S, D = y      # unpack current values of y
    rI, rR, dI, ks, ka, q = params  # unpack parameters
    derivs = [rI*I*S*(q*ks+(1-q)*ka)-I*rR-I*dI,      # list of dy/dt=f functions
             I*rR-rI*I*S*(q*ks+(1-q)*ka),
             I*dI]
    return derivs


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

# Bundle initial conditions for ODE solver
y0 = [I0, S0, D0]

# Make time array for solution
tStop = 200.
tInc = 0.5
t = np.arange(0., tStop, tInc)

# Call the ODE solver

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1(children='Corona - Ratenmodell'),

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
     Input('percentage-symptomatic-slider', 'value'),])
def update_figure(infection_rate, recovery_rate, death_rate, symptomatic, asymptomatic, percentage_symptomatic):
    params = [infection_rate, recovery_rate, death_rate, symptomatic, asymptomatic, percentage_symptomatic]
    psoln = odeint(f, y0, t, args=(params,))

    return {
        'data': [
             {'x': t, 'y': psoln[:, 0], 'type': 'line', 'name': 'Infizierte'},
             {'x': t, 'y': psoln[:, 1], 'type': 'line', 'name': 'Gesunde'},
             {'x': t, 'y': psoln[:, 2], 'type': 'line', 'name': 'Verstorbene'},
        ],
        'layout': dict(
            xaxis={'title': 'Zeit'},
            yaxis={'title': 'Prozent', 'range': [0, 100]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            transition={'duration': 500},
        )
    }


app.run_server(debug=False)
