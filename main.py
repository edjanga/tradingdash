import plotly.graph_objects as go
import plotly.express as px
from DataStore import Data
from dash import Dash, Output, Input, dcc, html
import pdb
import pandas as pd

app = Dash(__name__)
data_obj = Data()


allocation_query = data_obj.write_query_allocation()
allocation_ls = data_obj.query(query=allocation_query).name.tolist()

app.layout = html.Div([dcc.RadioItems(id='allocation',options=allocation_ls,value=allocation_ls[0]),\
                      html.Div(id='layout',children=[])])

@app.callback(
    Output(component_id='layout',component_property='children'),
    Input(component_id='allocation',component_property='value')
)
def strategies_dropdown(allocation):
    content_ls = []
    query = data_obj.write_query_equity_curves(allocation=allocation)
    equity_curves_aggregate_df = data_obj.query(query=query,melt=True).rename(columns={'index':'time'})
    equity_curves_fig = px.line(data_frame=equity_curves_aggregate_df,
                  x='time',\
                  y='equity_curve',color='strategy')
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
    perf_fig.update_layout(margin={'t': 30, 'b': 10},height=500)
    content_ls.append(dcc.Graph(figure=perf_fig))
    #pdb.set_trace()
    return content_ls




if __name__ == '__main__':
    app.run_server(debug=True)

