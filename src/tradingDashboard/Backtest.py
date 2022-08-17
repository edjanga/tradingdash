import pandas as pd
import numpy as np
import concurrent.futures
from src.tradingDashboard.data import Data
import inspect
from scipy.stats import skew,kurtosis
import pdb
#from empyrical import conditional_value_at_risk

data_obj = Data()

class Performance:

    """
        Backtester environment allowing one to compute performance metrics for a given dataframe
        where:
            -columns: strategies
            -row: date
            -data: returns
    """
    @staticmethod
    def returns(df):
        t = (df.index[-1] - df.index[0]).days / 365.25
        return (df+1).apply(lambda x,t: (x[-1] / x[0]) ** (1 /t) - 1,args=[t])

    @staticmethod
    def vol(df):
        if isinstance(df,pd.DataFrame):
            return pd.DataFrame(df.std())
        if isinstance(df,pd.Series):
            return df.std()

    @staticmethod
    def returns_adjusted(df):
        query = data_obj.write_query_symbol(symbol=['BIL'])
        rf_df = data_obj.query(query,set_index=True).apply(lambda x: np.log(x/x.shift())).dropna()
        rf_df.index.name = 'time'
        if isinstance(df,pd.DataFrame):
            df_copy = df.iloc[1:,:].subtract(rf_df.values, 1)
        if isinstance(df,pd.Series):
            df_copy = df[1:] - rf_df.BIL
        return df_copy

    @staticmethod
    def sharpe(df):
        if isinstance(df,pd.DataFrame):
            df_copy = Performance.returns_adjusted(df)
            return pd.DataFrame(pd.DataFrame(df_copy.mean()/df_copy.std()))
        if isinstance(df,pd.Series):
            df_copy = Performance.returns_adjusted(df)
            return df_copy.mean()/df_copy.std()

    @staticmethod
    def cvar(df):
        cvar_ls = list(map(lambda x: x[x<=np.quantile(x,.05)].mean(), df.transpose().values))
        return pd.DataFrame(cvar_ls,index=df.columns)

    @staticmethod
    def maxdrawdown(df):
        if isinstance(df,pd.Series):
            df_copy = pd.DataFrame(df+1)
            df_copy = df_copy.div(df_copy.cummax()) - 1
            max_drawdown_s = df_copy.cummin().min()
            return max_drawdown_s
        if isinstance(df,pd.DataFrame):
            df_copy = df+1
            df_copy = df_copy.div(df_copy.cummax()) - 1
            max_drawdown_s = df_copy.cummin().min()
            return pd.DataFrame(max_drawdown_s)

    @staticmethod
    def skew(df):
        skew_ls = list(map(lambda x: skew(x),df.transpose().values))
        return pd.DataFrame(skew_ls,index=df.columns)

    @staticmethod
    def kurtosis(df):
        kurtosis_ls = list(map(lambda x: kurtosis(x), df.transpose().values))
        return pd.DataFrame(kurtosis_ls,index=df.columns)

    @staticmethod
    def rolling_maxdrawdown(df,rolling_period=12):
        return np.sqrt(rolling_period)*df.rolling(rolling_period).apply(Performance.maxdrawdown).dropna()

    @staticmethod
    def rolling_sharpe(df,rolling_period=12):
        df_copy = Performance.returns_adjusted(df)
        rolling_obj = df_copy.rolling(rolling_period)
        df_copy = (rolling_obj.mean()).div(rolling_obj.std()).dropna()
        return np.sqrt(rolling_period)*df_copy

    @staticmethod
    def rolling_vol(df,rolling_period=12):
        return np.sqrt(rolling_period)*df.rolling(rolling_period).std().dropna()


class Table(Performance):

    def __init__(self,df):
        self.df = df
        methods_ls = inspect.getmembers(Performance,predicate=inspect.isfunction)
        self.metric_name = [method[-1] for method in methods_ls if ('rolling' not in method[0])&('adjusted' not in method[0])]
        self.rolling_name = [method[-1] for method in methods_ls if ('rolling' in method[0])&('adjusted' not in method[0])]

    def table_aggregate(self):

        aggregate_perf_dd = dict()

        def add_performance_metrics(metric,df):
            aggregate_perf_dd[metric.__name__] = metric(df)
            return aggregate_perf_dd

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for metric in self.metric_name:
                results.append(executor.submit(add_performance_metrics,metric,self.df))
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
        return aggregate_roll_dd


if __name__ == '__main__':
    allocation = 'buy_and_hold'
    data_obj = Data()
    perf_obj = Performance()
    query = data_obj.write_query_returns(allocation)
    df = data_obj.query(query,set_index=True)
    df.index.name = 'time'
    perf_obj.rolling_sharpe(df)
    table_obj = Table(df)
    table_obj.returns(df)
    aggregate_perf_df = table_obj.table_aggregate()
    print(aggregate_perf_df.head())