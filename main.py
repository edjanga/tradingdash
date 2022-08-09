import plotly.graph_objects as go
import plotly.express as px
from DataStore import Data
from dash import Dash, Output, Input, dcc, html
import pdb

app = Dash(__name__)
data_obj = Data()

allocation_query = data_obj.write_query_allocation()
allocation_ls = data_obj.query(query=allocation_query)
pdb.set_trace()

#app.layout = html.Div(dcc.RadioItems())