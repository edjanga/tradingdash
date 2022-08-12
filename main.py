import plotly.graph_objects as go
import plotly.express as px
from DataStore import Data
from dash import Dash, Output, Input, dcc, html
from plotly.subplots import make_subplots
import os
import pickle
import pdb
import pandas as pd
import sqlite3 as sql

app = Dash(__name__)
data_obj = Data()

strategy_dd = {'buy_and_hold':'BuyAndHold',
               'tactical_allocation':'TacticalAllocation'}
path_pickle = os.path.abspath('/Users/emmanueldjanga/wifeyAlpha/DataStore')
allocation_query = data_obj.write_query_allocation()
allocation_ls = data_obj.query(query=allocation_query).name.tolist()

app.layout = html.Div([dcc.RadioItems(id='allocation',options=allocation_ls,value=allocation_ls[0]),\
                       html.Div(id='layout_aggregate',children=[]),\
                       html.Br(),\
                       html.Br(),\
                       dcc.Dropdown(id='strategy',value='average'),\
                       dcc.Graph(id='layout_individual')])

@app.callback(
    Output(component_id='layout_aggregate',component_property='children'),
    Input(component_id='allocation',component_property='value')
)
def aggregate_layout(allocation):
    content_ls = []
    query = data_obj.write_query_equity_curves(allocation=allocation)
    #equity_curves_aggregate_df = data_obj.query(query=query,melt=True).rename(columns={'index':'time'})
    equity_curves_aggregate_df = data_obj.query(query=query, melt=True,set_index=True).rename(index={'index': 'time'})
    #pdb.set_trace()
    # equity_curves_fig = px.line(data_frame=equity_curves_aggregate_df,
    #               x='time',\
    #               y='equity_curve',color='strategy')
    equity_curves_fig = px.line(data_frame=equity_curves_aggregate_df,y='equity_curve', color='strategy')
    equity_curves_fig.add_hline(y=1)
    content_ls.append(html.Br())
    content_ls.append(dcc.Graph(figure=equity_curves_fig))
    query = data_obj.write_query_returns(allocation=allocation)
    strategy_returns_aggregate_df = data_obj.query(query=query,\
                                                   melt=True).rename(columns={'equity_curve':'returns'}).set_index('index')
    strategy_returns_aggregate_df = strategy_returns_aggregate_df.rename(columns={'index':'time'})
    strategy_returns_aggregate_df.index.name = None
    distribution_returns_fig = px.histogram(data_frame=strategy_returns_aggregate_df,\
                                            y='returns',color='strategy',labels={'x':'frequency'},marginal='violin')
    content_ls.append(html.Br())
    content_ls.append(dcc.Graph(figure=distribution_returns_fig))
    query = data_obj.write_query_performance(allocation=allocation)
    perf_df = data_obj.query(query).rename(columns={'index':'strategy'})
    perf_fig = go.Figure(data=[go.Table( \
        header=dict(values=list(perf_df.columns), align='left'), \
        cells=dict(values=perf_df.transpose().values, align='left'))])
    perf_fig.update_layout(margin={'t': 30, 'b': 10},height=300)
    content_ls.append(dcc.Graph(figure=perf_fig))
    return content_ls

@app.callback(
    Output(component_id='strategy',component_property='options'),
    Input(component_id='allocation',component_property='value')
)
def strategies_dropdown(allocation):
    query = data_obj.write_query_strategies(allocation)
    strategies_ls = data_obj.query(query)['name'].tolist()
    strategies_ls.remove('index')
    return strategies_ls

@app.callback(
    Output(component_id='layout_individual',component_property='figure'),
    [Input(component_id='allocation',component_property='value'),\
     Input(component_id='strategy',component_property='value')]
)
def strategy_layout(allocation,strategy):
    f = '/'.join((path_pickle,'.'.join((strategy_dd[allocation],'pickle'))))
    pickle_in = open(f,'rb')
    rolling_perf_dd = pickle.load(pickle_in)
    pickle_in.close()
    rolling_perf_df = pd.DataFrame()
    for key,value in rolling_perf_dd.items():
        rolling_perf_df = pd.concat([rolling_perf_df,value[[strategy]].rename(columns={strategy:key})],axis=1)
    rolling_perf_metrics_ls = rolling_perf_df.columns.tolist()
    fig = make_subplots(rows=len(rolling_perf_metrics_ls),cols=1,shared_xaxes=True,x_title='time')
    for i,col in enumerate(rolling_perf_metrics_ls):
        if col == 'rolling_maxdrawdown':
            fig.add_trace(go.Scatter(x=rolling_perf_df.index,\
                                     y=rolling_perf_df[col],\
                                     mode='lines',name=f'{col}',fill='tozeroy'),row=i+1,col=1)
        else:
            fig.add_trace(go.Scatter(x=rolling_perf_df.index, \
                                     y=rolling_perf_df[col], \
                                     mode='lines', name=f'{col}'),row=i+1,col=1)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)

