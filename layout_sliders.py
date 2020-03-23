import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from lmfit import Parameters

from get_data import get_corona_data
from logistic_model import logistic_function, LogisticModel

COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# server = app.server

app.layout = dbc.Container(
    children=[
        dcc.Store(id='corona-data', ),
        dcc.Store(id='population-data'),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Button('Daten neu laden.', id='reload-button'), width='auto'),
                        dbc.Col(dcc.Dropdown(
                            options=[{'label': country, 'value': country} for country in ('Germany', 'France')],
                            id='country-dropdown',
                            value='Germany',
                        )),
                        dbc.Col(html.Button('Daten fitten', id='fit-button'), width='auto')
                    ]
                ),
                dcc.Graph(id='figure'),
                dbc.Row(
                    [
                        dbc.Col(dbc.ButtonGroup(
                            [
                                dbc.Button("Linear"),
                                dbc.Button("Logarithmisch")
                            ]
                        ), width='auto'),
                        html.Label(id='range-label'),
                        dbc.Col(dcc.RangeSlider(
                            min=0,
                            max=200,
                            value=[0, 200],
                            id='range-slider'
                        ))
                    ]
                ),
            ], style={"border": "1px black solid", "border-radius": "5px", "border-color": "grey",
                      "padding": "12px 25px 25px 25px"}
        ),
        html.Div(
            [
                html.H3("Verbreitung des Virus"),
                html.P(r"Die Infectionsrate ergibt sich aus der durchschnittlichen Anzahl an Kontakten <k>"
                       r" und der Wahrscheinlichkeit einer Ansteckung p. Ist die Infektionsrate"
                       r" größer als die Genesungsrate, verbreitet sich das Virus exponentiell."),
                html.Label(id='infection-label'),
                dcc.Slider(
                    id='infection-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
                    value=0.1,
                    disabled=True
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.FormGroup(
                                [
                                    dbc.Label(id='infection_probability-label'),
                                    dcc.Slider(
                                        id='infection_probability-slider',
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
                                        value=0.1,
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            dbc.FormGroup(
                                [
                                    html.Label(id='number_contact-label'),
                                    dcc.Slider(
                                        id='number_contact-slider',
                                        min=0,
                                        max=20,
                                        step=0.1,
                                        marks={np.round(i, 1): str(np.round(i, 1)) for i in np.arange(0, 20, 1.)},
                                        value=2,
                                    ),
                                ]
                            )
                        )
                    ]
                ),
                html.Label('Anfänglich Infizierte:'),
                dcc.Input(
                    id='initial-input',
                    type='number',
                    value=10
                )
            ], style={"border": "1px black solid", "border-radius": "5px", "border-color": "grey",
                      "padding": "12px 25px 25px 25px"}
        ),
        html.Div(
            [
                html.H3("Rückgang des Virus"),
                html.P("Sowohl Genesene als auch Gestorbene können das Virus nicht weiter verbreiten. Für die "
                       "Ansteckung neuer infizierter ist also eine effektive Rückgangsrate relevant, die sich aus der "
                       "Genesungsrate und Mortalität ergibt."),
                dbc.Label(id='reduction-label'),
                dcc.Slider(
                    id='reduction-slider',
                    min=0,
                    max=1,
                    step=0.001,
                    marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
                    disabled=True
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.FormGroup(
                                [
                                    dbc.Label(id='recovery-label'),
                                    dcc.Slider(
                                        id='recovery-slider',
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        marks={np.round(i, 2): str(np.round(i, 2)) for i in np.arange(0, 1, 0.1)},
                                        value=0.5,
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            dbc.FormGroup(
                                [
                                    dbc.Label(id='mortality-label'),
                                    dcc.Slider(
                                        id='mortality-slider',
                                        min=0,
                                        max=10,
                                        step=0.1,
                                        marks={i: str(i) + "%" for i in range(11)},
                                        # marks={np.round(i, 1): str(np.round(i, 1)) for i in np.arange(0., 11., 1.)},
                                        value=1,
                                    )
                                ]
                            )
                        )
                    ]
                )
            ], style={"border": "1px black solid", "border-radius": "5px", "border-color": "grey",
                      "padding": "12px 25px 25px 25px"}
        ),
    ])


@app.callback(
    Output('corona-data', 'data'),
    [Input('reload-button', 'n_clicks')],
    [State('corona-data', 'data')]
)
def load_data(clicks, corona_data):
    if clicks is None:
        raise PreventUpdate
    data_country = get_corona_data().reset_index()
    return data_country.to_json()


@app.callback(
    Output('population-data', 'data'),
    [Input('reload-button', 'n_clicks')],
    [State('population-data', 'data')]
)
def load_population_data(clicks, data):
    if clicks is None:
        raise PreventUpdate
    population_url = 'https://raw.githubusercontent.com/datasets/population/master/data/population.csv'
    population = pd.read_csv(population_url)
    population = population.set_index(['Country Name', 'Year']).Value
    return population.unstack().iloc[:, -1].to_json()


@app.callback(
    [Output('country-dropdown', 'options'),
     Output('range-slider', 'value')],
    [Input('corona-data', 'modified_timestamp')],
    [State('corona-data', 'data')]
)
def update_country_dropdown(ts, data):
    if ts is None:
        raise PreventUpdate
    data = pd.read_json(data)
    countries = data['Country/Region'].unique()

    data = data.set_index('Country/Region', inplace=False).loc['Germany']
    data = data.set_index('Date')
    dates = data.index.dayofyear - data.index.dayofyear[0]
    return [{'label': country, 'value': country} for country in countries], (dates.min(), dates.max())


@app.callback(
    Output('range-label', 'children'),
    [Input('range-slider', 'value')]
)
def update_label(values):
    return 'Fit Bereich: Tag {} bis Tag {}'.format(values[0], values[1])


@app.callback(
    [Output('infection_probability-slider', 'value'),
     Output('recovery-slider', 'value'),
     Output('initial-input', 'value')],
    [Input('fit-button', 'n_clicks'),
     Input('range-slider', 'value')],
    [State('country-dropdown', 'value'),
     State('number_contact-slider', 'value'),
     State('mortality-slider', 'value'),
     State('corona-data', 'data'),
     State('population-data', 'data')]
)
def fit_data(clicks, fit_range, country,
             number_contact, mortality,
             corona_data, population_data):
    if clicks is None:
        raise PreventUpdate
    data = pd.read_json(corona_data)
    data_country = data.set_index('Country/Region').loc[country]
    data_country = data_country.set_index('Date')
    dates = data_country.index.dayofyear - data_country.index.dayofyear[0]
    data_country.index = dates
    data_country = data_country.iloc[(data_country.index >= fit_range[0]) * (data_country.index <= fit_range[1])]
    population = pd.read_json(population_data, typ='series').loc[country]

    model = LogisticModel
    logistic_params = Parameters()
    logistic_params.add('infection_rate', value=0.01, min=0)
    logistic_params.add('recover_rate', value=0.01, max=1)
    logistic_params.add('infected_0', value=1e-3, min=0, max=1)
    logistic_params.add('mortality', value=mortality/100, min=0, max=1, vary=False)
    fit = model.fit(data_country['Active Cases']/population, params=logistic_params, time=data_country.index,
                    method='powell')
    infected_0 = fit.eval(params=fit.params, time=[-fit_range[0]])[0]
    print(infected_0, fit.best_values['infected_0'])
    return (fit.best_values['infection_rate']/number_contact,
            fit.best_values['recover_rate']/(1 + mortality/100),
            infected_0 * population)


@app.callback(
    Output('infection-label', 'children'),
    [Input('infection-slider', 'value')]
)
def update_infection(value):
    return 'Infectionsrate: {}'.format(round(value, 2))


@app.callback(
    Output('infection-slider', 'value'),
    [Input('infection_probability-slider', 'value'),
     Input('number_contact-slider', 'value')]
)
def update_infection(probability, number):
    return probability * number


@app.callback(
    Output('number_contact-label', 'children'),
    [Input('number_contact-slider', 'value')]
)
def update_infection(value):
    return 'Durchschnittliche anzahl Kontakte pro Tag: {}'.format(round(value, 1))


@app.callback(
    Output('infection_probability-label', 'children'),
    [Input('infection_probability-slider', 'value')]
)
def update_infection(value):
    return 'Wahrscheinlichkeit einer Ansteckung pro Kontakt: {}'.format(round(value, 2))


@app.callback(
    Output('reduction-slider', 'value'),
    [Input('recovery-slider', 'value'),
     Input('mortality-slider', 'value')]
)
def calculate_reduction(recovery, mortality):
    return recovery * (1 + mortality/100)


@app.callback(
    Output('reduction-label', 'children'),
    [Input('reduction-slider', 'value')]
)
def update_infection(value):
    return 'Reductionsrate: {}'.format(round(value, 2))


@app.callback(
    Output('recovery-label', 'children'),
    [Input('recovery-slider', 'value')]
)
def update_infection(value):
    return 'Genesungsrate: {}'.format(round(value, 2))


@app.callback(
    Output('mortality-label', 'children'),
    [Input('mortality-slider', 'value')]
)
def update_infection(value):
    return 'Mortalität: {}%'.format(round(value, 1))


@app.callback(
    Output('figure', 'figure'),
    [Input('reload-button', 'n_clicks'),
     Input('infection-slider', 'value'),
     Input('recovery-slider', 'value'),
     Input('mortality-slider', 'value'),
     Input('initial-input', 'value'),
     Input('corona-data', 'modified_timestamp'),
     Input('population-data', 'modified_timestamp'),
     Input('country-dropdown', 'value')],
    [State('range-slider', 'value'),
     State('corona-data', 'data'),
     State('population-data', 'data')]
)
def update_figure(clicks, infection_rate, recovery_rate, mortality, infected_0,
                  corona_ts, population_ts, country, fit_range, corona_data, population_data):
    if corona_ts is None:
        raise PreventUpdate
    data = pd.read_json(corona_data)
    data_country = data.set_index('Country/Region').loc[country]
    data_country = data_country.set_index('Date')
    dates = data_country.index.dayofyear - data_country.index.dayofyear[0]
    population = pd.read_json(population_data, typ='series').loc[country]
    dates_fit = np.linspace(fit_range[0], fit_range[1], 50)

    infected, recovered, deaths = logistic_function(
        dates_fit, infection_rate=infection_rate, recover_rate=recovery_rate, mortality=mortality/100,
        infected_0=infected_0/population
    )

    return {
        'data': [
             {'x': dates_fit, 'y': population * infected, 'type': 'line', 'name': 'Infizierte'},
             {'x': dates_fit, 'y': population * recovered, 'type': 'line', 'name': 'Genesene'},
             {'x': dates_fit, 'y': population * deaths, 'type': 'line', 'name': 'Verstorbene'},
             go.Scatter(x=dates, y=data_country['Active Cases'], name='Infizierte',
                        mode='markers', marker_color=COLORS[0]),
             go.Scatter(x=dates, y=data_country['Total Recoveries'], name='Genesene',
                        mode='markers', marker_color=COLORS[1]),
             go.Scatter(x=dates, y=data_country['Total Deaths'], name='Verstorbene',
                        mode='markers', marker_color=COLORS[2])
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
