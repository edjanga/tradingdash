import empyrical as ep
import pandas as pd
import numpy as np
import concurrent.futures
from DataStore import Data
import inspect
from Logger import Logs
from scipy.stats import skew,kurtosis
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

    @staticmethod
    def annualised_cagr(df,freq='monthly'):
        return pd.DataFrame(ep.cagr(df,period=freq),index=df.columns)

    @staticmethod
    def cvar(df):
        cvar_ls = list(map(lambda x:ep.conditional_value_at_risk(x),df.transpose().values))
        return pd.DataFrame(cvar_ls,index=df.columns)

    @staticmethod
    def annualised_calmar_ratio(df,freq='monthly'):
        calmar_ratio_ls = list(map(lambda x:ep.calmar_ratio(x,period=freq),df.transpose().values))
        return pd.DataFrame(calmar_ratio_ls,index=df.columns)

    @staticmethod
    def maxdrawdon(df):
        maxdrawdowns_ls = list(map(lambda x: ep.max_drawdown(x),df.transpose().values))
        return pd.DataFrame(maxdrawdowns_ls,index=df.columns)

    @staticmethod
    def skew(df):
        skew_ls = list(map(lambda x: skew(x),df.transpose().values))
        return pd.DataFrame(skew_ls,index=df.columns)

    @staticmethod
    def kurtosis(df):
        kurtosis_ls = list(map(lambda x: skew(x), df.transpose().values))
        return pd.DataFrame(kurtosis_ls, index=df.columns)

    @staticmethod
    def tail_ratio(df):
        tail_ratio_ls = list(map(lambda x: ep.tail_ratio(x), df.transpose().values))
        return pd.DataFrame(tail_ratio_ls, index=df.columns)

    @staticmethod
    def rolling_maxdrawdown(df,window=3):
        roll_maxdrawdown_ls = list(map(lambda x: ep.roll_max_drawdown(x,window), df.transpose().values))
        return pd.DataFrame(roll_maxdrawdown_ls,index=df.columns,columns=df.index[window-1:]).transpose()

    @staticmethod
    def rolling_sharpe(df,window=3):
        roll_sharpe_ls = list(map(lambda x: ep.roll_sharpe_ratio(x,window=window), df.transpose().values))
        return pd.DataFrame(roll_sharpe_ls,index=df.columns,columns=df.index[window - 1:]).transpose()

    @staticmethod
    def rolling_annual_vol(df,window=3):
        roll_annual_vol_ls = list(map(lambda x: ep.roll_annual_volatility(x,window=window),df.transpose().values))
        return pd.DataFrame(roll_annual_vol_ls,index=df.columns,columns=df.index[window - 1:]).transpose()


class Table(Performance):

    log_obj = Logs()

    def __init__(self,df):
        self.df = df
        methods_ls = inspect.getmembers(Performance,predicate=inspect.isfunction)
        self.metric_name = [method[-1] for method in methods_ls if 'rolling' not in method[0]]
        self.rolling_name = [method[-1] for method in methods_ls if 'rolling' in method[0]]

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

    def rolling_aggregate(self):

        aggregate_roll_dd = dict()

        def add_rolling_metrics(rolling_metric,df):
            aggregate_roll_dd[rolling_metric.__name__] = rolling_metric(df)

            return aggregate_roll_dd

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for metric in self.rolling_name:
                results.append(executor.submit(add_rolling_metrics,metric,self.df))
            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[COMPUTATION]: {f.__name__} metric is being computed @ \
                            {Table.log_obj.now_date()}.\n'
                    Table.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        return aggregate_roll_dd


if __name__ == '__main__':
    allocation = 'buy_and_hold'
    data_obj = Data()
    perf_obj = Performance()
    query = data_obj.write_query_returns(allocation)
    df = data_obj.query(query).set_index('index')
    df.index.name = 'time'
    #perf_obj.rolling_maxdrawdown(df,6)
    table_obj = Table(df)
    aggregate_perf_dd = table_obj.rolling_aggregate()
    print(aggregate_perf_dd.keys())