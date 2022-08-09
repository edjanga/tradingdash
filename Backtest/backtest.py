import empyrical as ep
import pandas as pd
import numpy as np
import concurrent.futures
from DataStore import Data
import inspect
from Logger import Logs
import pdb

class Performance:
    """
        Backtester environment allowing one to compute performance metrics for a given dataframe
        where:
            -columns: strategies
            -row: date
            -data: returns
    """
    @staticmethod
    def annualised_returns(df,freq='monthly'):
        return pd.DataFrame(ep.annual_return(df,period=freq))

    @staticmethod
    def annualised_vol(df,freq='monthly'):
        return pd.DataFrame(ep.annual_volatility(df,period=freq),index=df.columns)

class Table(object):

    log_obj = Logs()

    def __init__(self,object,df):
        self.df = df
        methods_ls = inspect.getmembers(object,predicate=inspect.isfunction)
        self.metric_name = [method[-1] for method in methods_ls]

    def table_aggregate(self):

        aggregate_perf_dd = dict()

        def add_performance_metrics(metric,df):
            aggregate_perf_dd[metric.__name__] = metric(df)
            return aggregate_perf_dd

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for metric in self.metric_name:
                results.append(executor.submit(add_performance_metrics,metric,self.df))
            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[COMPUTATION]: {f.__name__} metric is being computed @ \
                            {Table.log_obj.now_date()}.\n'
                    Table.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        aggregate_perf_df = pd.concat(aggregate_perf_dd, axis=1).droplevel(1, 1).round(4)
        return aggregate_perf_df


if __name__ == '__main__':
    allocation = 'buy_and_hold'
    data_obj = Data()
    perf_obj = Performance()
    query = data_obj.write_query_returns(allocation)
    df = data_obj.query(query).set_index('index')
    df.index.name = 'time'
    table_obj = Table(perf_obj, df)
    aggregate_perf_df = table_obj.table_aggregate()