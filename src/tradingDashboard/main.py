import plotly.graph_objects as go
import plotly.express as px
from src.tradingDashboard.data import Data
from dash import Dash, Output, Input, dcc, html
from plotly.subplots import make_subplots
from src.tradingDashboard.Strategies import BuyAndHold, TacticalAllocation, PortfolioStrategies
from src.tradingDashboard.Backtest import Table
import os
import pickle
import pandas as pd
from pathlib import Path
import asyncio
import concurrent.futures
import time


data_obj = Data()
async def insert_historical_data():
    data_obj.insert_historical_data()
async def update(allocations=['buy_and_hold', 'tactical_allocation']):
    """
        First insert historical data
        Then update all trading strategies (including performance metrics)
        Wait for one hour before the next update
    """
    while True:
        await insert_historical_data()
        query = data_obj.write_query_price()
        df = data_obj.query(query, set_index=True)
        for allocation in allocations:
            if allocation == 'buy_and_hold':
                allocation_obj = BuyAndHold()
            if allocation == 'tactical_allocation':
                allocation_obj = TacticalAllocation()
            table_returns = '_'.join((allocation, 'returns'))
            table_performance = '_'.join((allocation, 'performance'))
            portfolio_strat_obj = PortfolioStrategies(allocation_obj, df)
            equity_curves_df = portfolio_strat_obj.equity_curves_aggregate()
            equity_curves_df['average'] = equity_curves_df.sum(axis=1) / equity_curves_df.shape[1]
            portfolio_strat_obj.insertion(equity_curves_df, allocation_obj, table=allocation)
            portfolio_strat_obj.insertion(equity_curves_df - 1, allocation_obj, table=table_returns)
            query = data_obj.write_query_returns(allocation=allocation)
            returns_df = data_obj.query(query, set_index=True)
            returns_df.index.name = 'time'
            perf_obj = Table(returns_df)
            perf_df = perf_obj.table_aggregate()
            portfolio_strat_obj.insertion(perf_df, allocation_obj, table=table_performance)
            rolling_perf_dd = perf_obj.rolling_aggregate()
            portfolio_strat_obj.to_pickle(rolling_perf_dd, allocation_obj)
        time.sleep(3600)

app = Dash(__name__)

strategy_dd = {'buy_and_hold':'BuyAndHold','tactical_allocation':'TacticalAllocation'}
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
    equity_curves_aggregate_df = data_obj.query(query=query, melt=True,set_index=True).rename(index={'index': 'time'})
    equity_curves_fig = px.line(data_frame=equity_curves_aggregate_df,y='equity_curve', color='strategy',\
                                title='Equity curves',labels={'equity_curves':'value'})
    equity_curves_fig.add_hline(y=1)
    content_ls.append(html.Br())
    content_ls.append(dcc.Graph(figure=equity_curves_fig))
    query = data_obj.write_query_returns(allocation=allocation)
    strategy_returns_aggregate_df = data_obj.query(query=query,\
                                                   melt=True).rename(columns={'equity_curve':'returns'}).set_index('index')
    strategy_returns_aggregate_df = strategy_returns_aggregate_df.rename(columns={'index':'time'})
    strategy_returns_aggregate_df.index.name = None
    distribution_returns_fig = px.histogram(data_frame=strategy_returns_aggregate_df,\
        y='returns',color='strategy',labels={'x':'frequency'},marginal='violin',title='Returns distribution')
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
    f = '.'.join((strategy_dd[allocation],'pickle'))
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
                                     mode='lines',name=f'{col}',fill='tozeroy',line=dict(color=col_dd[col])),\
                          row=i+1,col=1)
        else:
            fig.add_trace(go.Scatter(x=rolling_perf_df.index, \
                                     y=rolling_perf_df[col], \
                                     mode='lines', name=f'{col}',line=dict(color=col_dd[col])),row=i+1,col=1)

    return fig


if __name__ == '__main__':

    def update_in_the_background():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(update())
        loop.close()

    def run_app_and_update():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for func in [app.run_server,update_in_the_background]:
                results.append(executor.submit(func))

    run_app_and_update()

