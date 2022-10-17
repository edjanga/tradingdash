import plotly.graph_objects as go
import plotly.express as px
from src.tradingDashboard.data import Data
from dash import Dash, Output, Input, dcc, html
from plotly.subplots import make_subplots
import os
import pickle
import pandas as pd
from pathlib import Path
import time
import pdb

data_obj = Data()

app = Dash(__name__)
server = app.server
strategy_dd = {'buy_and_hold':'BuyAndHold',
               'tactical_allocation':'TacticalAllocation',
               'cross_asset_allocation':'CrossAllocation'}
root = Path(__file__).parent
path_pickle = os.path.relpath(path='DataStore',start=root)
allocation_query = data_obj.write_query_allocation()
allocation_ls = data_obj.query(query=allocation_query).name.tolist()
app.layout = html.Div([html.H1('Asset Allocation Baskets - Dashboard',style={'text-align':'center','padding': '10px'}),\
                       dcc.RadioItems(id='allocation',options=allocation_ls,value=allocation_ls[0],\
                                      inputStyle={'margin': '10px'},style={'text-align': 'left'}),\
                       html.H2('Equity curves, returns distribution and performance metrics',\
                               style={'text-align':'left'}),\
                       html.Div(id='layout_aggregate',children=[]),\
                       html.Br(),\
                       html.Br(),\
                       html.H2('Indivdual strategy',style={'text-align':'left'}),\
                       dcc.Dropdown(id='strategy',value='average'),\
                       dcc.Graph(id='layout_individual')],style={'font-family':'verdana','margin': '20px'})


@app.callback(
    Output(component_id='layout_aggregate',component_property='children'),
    Input(component_id='allocation',component_property='value')
)
def aggregate_layout(allocation):
    content_ls = []
    query = data_obj.write_query_equity_curves(allocation=allocation)
    equity_curves_aggregate_df = data_obj.query(query=query, melt=True,set_index=True)
    equity_curves_aggregate_df.index.name = 'time'
    equity_curves_fig = px.line(data_frame=equity_curves_aggregate_df,y='equity_curve', color='strategy',\
                                title='Equity curves',labels={'equity_curve':'value'})
    equity_curves_fig.add_hline(y=1)
    content_ls.append(html.Br())
    content_ls.append(dcc.Graph(figure=equity_curves_fig))
    # Histogram
    query = data_obj.write_query_returns(allocation=allocation)
    strategy_returns_aggregate_df = data_obj.query(query=query, \
                                                   melt=True).rename(columns={'equity_curve': 'returns'})
    strategy_returns_aggregate_df = strategy_returns_aggregate_df.rename(columns={'index':'time'})
    strategy_returns_aggregate_df.index.name = None
    distribution_returns_fig = px.histogram(data_frame=strategy_returns_aggregate_df,\
        y='returns',color='strategy',labels={'x':'frequency'},marginal='violin',title='Returns distribution')
    content_ls.append(html.Br())
    content_ls.append(dcc.Graph(figure=distribution_returns_fig))
    # Table
    query = data_obj.write_query_performance(allocation=allocation)
    perf_df = data_obj.query(query).rename(columns={'index':'strategy'})
    perf_df.iloc[:,1:] = perf_df.iloc[:,1:].astype(float).round(4)
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
    f = Path(f'src/tradingDashboard/{".".join((strategy_dd[allocation],"pickle"))}')
    pickle_in = open(f,'rb')
    rolling_perf_dd = pickle.load(pickle_in)
    pickle_in.close()
    rolling_perf_df = pd.DataFrame()
    for key,value in rolling_perf_dd.items():
        rolling_perf_df = pd.concat([rolling_perf_df,value[[strategy]].rename(columns={strategy:key})],axis=1)
    rolling_perf_metrics_ls = rolling_perf_df.columns.tolist()
    fig = make_subplots(rows=len(rolling_perf_metrics_ls),cols=1,shared_xaxes=True,x_title='time',\
                        column_titles=['Rolling metrics'])
    col_dd = {'rolling_maxdrawdown':'red','rolling_vol':'blue','rolling_sharpe':'green'}
    for i,col in enumerate(rolling_perf_metrics_ls):
        if col == 'rolling_maxdrawdown':
            fig.add_trace(go.Scatter(x=rolling_perf_df.index,\
                                     y=rolling_perf_df[col],\
                                     mode='lines',name=f'{col} - 6 months',fill='tozeroy',line=dict(color=col_dd[col])),\
                          row=i+1,col=1)
        else:
            fig.add_trace(go.Scatter(x=rolling_perf_df.index, \
                                     y=rolling_perf_df[col], \
                                     mode='lines', name=f'{col} - 6 months',line=dict(color=col_dd[col])),row=i+1,col=1)

    return fig


if __name__ == '__main__':

    app.run_server(port=8051)



