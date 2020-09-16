# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:01:14 2019

@author: rcloke
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

from ILMN_test import modelTest



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

run_model = modelTest('ILMN')
df = run_model[0]
tomorrow_price = run_model[1]

app.layout = html.Div([
    html.H1(children='Stock Market Predictions Using Machine Learning'),
    
    html.Div(children='''
        Select a stock to fetch data and run real-time predictions using a pre-trained machine learning model
    '''),
             
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'ILMN', 'value': 'ILMN'},
            {'label':'INTC','value':'INTC'},
            {'label':'JNJ','value':'JNJ'}
        ],
        value='ILMN'
        ),
    html.Div(id='output-container'),
            
            
    dcc.Graph(
        id='graph-model',
        figure={
            'data': [
                go.Scatter(
                    x= df.index.to_series(),
                    y=df['Close'],
                    opacity=0.7,
                    name='actual close',
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'blue'}
                    },
                ),
                go.Scatter(
                    x= df.index.to_series(),
                    y=df['pred'],
                    opacity=0.7,
                    name='predicted close',
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'orange'}
                    },
                ),
            ],
            'layout': go.Layout(
                xaxis={'title': 'Stock price by date','dtick':1},
                yaxis={'title': 'Stock price'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
    
    dcc.Textarea(
    id = 'text-box',
    placeholder='Enter a value...',
    value='test',
    style={'width': '100%'}
    ),
    html.Div(children='''
        contact: ryan.cloke@gmail.com
    ''')
])

@app.callback(
    Output(component_id='text-box', component_property='value'),
    [Input(component_id='my-dropdown', component_property='value')]
)
def update_output_div(input_value):
    tomorrow_price = modelTest(input_value)[1]
    return 'Tomorrow\'s predicted closing price is "{}"'.format(tomorrow_price)

@app.callback(
    Output('graph-model', 'figure'),
    [Input('my-dropdown', 'value')])


def update_figure(selected_stock):
    run_model = modelTest(selected_stock)
    df = run_model[0]
    #tomorrow_price = run_model[1]

    return {
        'data': [
                    go.Scatter(
        x= df.index.to_series(),
        y=df['Close'],
        opacity=0.7,
        name='actual close',
        marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'blue'}
                },
        ),
    go.Scatter(
        x= df.index.to_series(),
        y=df['pred'],
        opacity=0.7,
        name='predicted close',
        marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'orange'}
                },
        )],
        'layout': go.Layout(
                xaxis={'title': 'Stock price by date'},
                yaxis={'title': 'Stock price'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
                )
    }
    


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False,threaded=True)