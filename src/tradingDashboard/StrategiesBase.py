import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay

"""
    Functions used in some strategy classes to generate signals
"""
def is_end_business_month(date):
    return (date.month + 1) == (date + BDay(1)).month
def momentum_score(df,close):
    p0 = df.copy()
    momentum_score_df = pd.DataFrame(index=df.index,columns=df.columns,data=0)
    for i in [1,3,6,12]:
        temp = close.resample('BM').last().resample('B').last().ffill().loc[:datetime.datetime.today(),:].shift(i)
        momentum_score_df = momentum_score_df+((12/i)*p0.div(temp))
    return momentum_score_df-19
def cov_matrix(df):
    p0 = df.copy()
    cov_matrix_df = pd.DataFrame(index=pd.MultiIndex.from_product([np.array(df.index),np.array(df.columns)]),columns=df.columns, data=0)
    month = 126
    for i in [1, 3, 6, 12]:
        cov_matrix_df += p0.rolling(window=i*month).corr()
    return cov_matrix_df/19
def momentum_score_sma(df,window=252):
    return df.div(df.rolling(window=window).mean())
def vol_estimate(df):
    return df.rolling(window=3*252).std()
def ri(df):
    month = int(252/12)
    average_returns_df = pd.DataFrame(index=df.index,columns=df.columns,data=0)
    for i in [1,3,6,12]:
        average_returns_df += np.log(df.div(df.shift(i*month)))
    return average_returns_df
def ci(df):
    corr_df = np.log(df.div(df.shift())).copy()
    corr_df['equally_weighted'] = corr_df.mean(axis=1)
    corr_df = corr_df.shift(252).expanding().corr()
    return corr_df
"""
    Base class
"""
class Strategy:
    """
        Class created to facilitates backtesting using vectorbt framework
    """
    def __init__(self,name,instruments,allocation,end=datetime.datetime.today()):
        start = end - relativedelta(years=25)
        self.__date_range = pd.bdate_range(start=start,end=end)[:-1]
        self.__name = name
        self.__instruments = instruments
        self.__allocation = allocation
"""
    Parent class for Buy and Hold strategies
"""
class BuyAndHold(Strategy):
    def __init__(self,instruments,name,holdings):
        super().__init__(instruments=instruments,allocation='buy_and_hold',name=name)
        self.holdings = holdings

    @property
    def instruments(self):
        return self._Strategy__instruments

    @property
    def name(self):
        return self._Strategy__name

    @property
    def date_range(self):
        return self._Strategy__date_range

    @property
    def allocation(self):
        return self._Strategy__allocation

    def rule(self, price):
        price_copy = price.loc[:,self.instruments].copy()
        universe_monthly_df = price_copy.resample('BM').last()
        universe_monthly_df = universe_monthly_df.iloc[:-1, :]
        first_valid_index_s = universe_monthly_df.apply(lambda x: x.first_valid_index()).sort_values()
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for instrument, date in first_valid_index_s.iteritems():
            signals.loc[date, instrument] = 1
        signals.replace(np.nan, 0, inplace=True)
        signals = signals.values
        return signals
"""
    Parent class for Tactical Asset Allocations strategies
"""
class TacticalAssetAllocation(Strategy):
    def __init__(self,instruments,name):
        super().__init__(instruments=instruments,allocation='tactical_asset_allocation',name=name)

    @property
    def instruments(self):
        return self._Strategy__instruments

    @property
    def name(self):
        return self._Strategy__name

    @property
    def date_range(self):
        return self._Strategy__date_range

    @property
    def allocation(self):
        return self._Strategy__allocation

    def rule(self, price):
        """
            To be overidden by each strategy
        """
        pass
