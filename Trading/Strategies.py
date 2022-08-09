import pandas as pd
import inspect
from DataStore import Data
import concurrent.futures
import pdb
from Logger import Logs
import numpy as np
from datetime import datetime
import empyrical as ep


data_obj = Data()

def returns(df):
    return df.pct_change()

def weights(weights_ls,returns_df):
    weights_ls = [weights_ls * returns_df.shape[0]]
    weights_ls = np.reshape(np.array(weights_ls), newshape=(returns_df.shape[0],returns_df.shape[1]))
    return weights_ls

def equity_curve(returns_df,weights_df):
    weights_df = weights_df.shift()
    equity_curve_df = pd.DataFrame((weights_df.mul(returns_df).sum(axis=1)+1).cumprod())
    return equity_curve_df
"""
    The following strategies were coded up according to @WifeyAlpha instructions.
    However, cash instrument has been replaced by JPST instead of BIL (BIL was used as cash money market
    instrument and bonds at the same time in certain strategies.
"""
class BuyAndHold:

    @staticmethod
    def golden_butterfly(df):
        """
            40% fixed income (20% SHY, 20% TLT)
            40% equities (20% VTI, 20% IWN)
            20% gold (GLD)
        """
        universe_df = df.filter(regex=r'(SHY|TLT|VTI|IWN|GLD)')
        returns_df = returns(universe_df)
        weights_ls = [.2] * returns_df.shape[1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df,weights_df)
        return equity_curve_df

    @staticmethod
    def rob_armott(df):
        """
            20% BNDX,20% LQD,10% VEU,10% VNQ,
            10% SPY,10% TLT,10% TIP,10% DBC
        """
        universe_df = df.filter(regex=r'(BNDX|LQD|VEU|VNQ|SPY|TLT|TIP|DBC)')
        returns_df = returns(universe_df)
        weights_ls = [.2,.2,.1,.1,.1,.1,.1,.1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def global_asset_allocation(df):
        """
            18% SPY,13.5% EFA,4.5% EEM, 19.8% LQD, 14.4% BNDX,
            13.5% TLT, 1.8% TIP, 5% DBC, 5% GLD, 4.5% VNQ
        """
        universe_df = df.filter(regex=r'(SPY|EFA|EEM|LQD|BNDX|TLT|TIP|DBC|GLD|VNQ)')
        returns_df = returns(universe_df)
        weights_ls = [.18,.135,.045,.198,.144,.135,.018,.05,.05,.045]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def permanent(df):
        """
            25% BIL,25% GLD,25% TLT and 25% SPY
        """
        universe_df = df.filter(regex=r'(BIL|GLD|TLT|SPY)')
        returns_df = returns(universe_df)
        weights_ls = [.25] * returns_df.shape[1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            pdb.set_trace()
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def desert(df):
        """
            60% IEF, 30% VTI, 10% GLD
        """
        universe_df = df.filter(regex=r'(IEF|VTI|GLD)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.3,.1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def larry(df):
        """
            30% in equities  (15% IWN, 7.5% IWN, 7.5% EEM)
            70% in bonds (IEF)
        """
        universe_df = df.filter(regex=r'(IWN|IWN|EEM|LQD|IEF)')
        returns_df = returns(universe_df)
        weights_ls = [.15,.075,.075,.7]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def big_rocks(df):
        """
            60% AGG, 6% SPY, 6% IWD, 6% IWM, 6% IWN, 4% EFV, 4% VNQ, 2% EFA, 2% SCZ, 2% DLS, 2% EEM
        """
        universe_df = df.filter(regex=r'(AGG|SPY|IWD|IWM|IWN|EFV|VNQ|EFA|SCZ|DLS|EEM)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.06,.06,.06,.06,.04,.04,.02,.02,.02,.02]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def sandwich(df):
        """
            50% in equities (20% SPY, 10% SCZ, 8% IWM, 6% EEM, 6% EFA)
            41% in bonds (41% IEF)
            9% in cash and REITs (5% VNQ, 4% JPST)
        """
        universe_df = df.filter(regex=r'(SPY|SCZ|IWM|EEM|EFA|IEF|VNQ|JPST)')
        returns_df = returns(universe_df)
        weights_ls = [.2,.1,.08,.06,.06,.41,.05,.04]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def balanced_tax_aware(df):
        """
            38% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM
        """
        universe_df = df.filter(regex=r'(AGG|SPY|BIL|EFA|IWM|VNQ|DBC|EEM)')
        returns_df = returns(universe_df)
        weights_ls = [.38,.15,.15,.13,.05,.05,.05,.04]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def balanced(df):
        """
            33% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM, 2% TIP, 2% BNDX, 1% HYG
        """
        universe_df = df.filter(regex=r'(AGG|SPY|BIL|EFA|IWM|VNQ|DBC|EEM|TIP|BNDX|HYG)')
        returns_df = returns(universe_df)
        weights_ls = [.33,.15,.15,.13,.05,.05,.05,.04,.02,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def income_growth(df):
        """
            37% AGG, 20% BIL, 10% TIP, 9% SPY, 8% EFA, 5% VNQ, 4% HYG, 4% BNDX, 2% IWM, 1% DBC
        """
        universe_df = df.filter(regex=r'(AGG|BIL|TIP|SPY|EFA|VNQ|HYG|BNDX|IWN|DBC)')
        returns_df = returns(universe_df)
        weights_ls = [.37,.2,.1,.09,.08,.05,.04,.04,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def income_growth_tax(df):
        """
            55% AGG, 20% BIL, 9% SPY, 8% EFA, 5% VNQ, 2% IWM, 1% DBC
        """
        universe_df = df.filter(regex=r'(AGG|BIL|SPY|EFA|VNQ|IWM|DBC)')
        returns_df = returns(universe_df)
        weights_ls = [.55,.2,.09,.08,.05,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def conservative_income(df):
        """
            70% in bonds (40% AGG, 18% BIL, 7% HYG, 5% BNDX),
            25% in cash (25% JPST), and
            5% in REITs (5% VNQ).
        """
        universe_df = df.filter(regex=r'(AGG|BIL|HYG|BNDX|JPST|VNQ)')
        returns_df = returns(universe_df)
        weights_ls = [.4,.18,.07,.05,.25,.05]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def conservative_income_tax(df):
        """
            70% in bonds (70% AGG),
            25% in cash (25% JPST), and
            5% in REITs (5% VNQ).
        """
        universe_df = df.filter(regex=r'(AGG|JPST|VNQ)')
        returns_df = returns(universe_df)
        weights_ls = [.7,.25,.05]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def all_weather(df):
        """
            30% SPY, 40% TLT, 15% IEF, 7.5% GLD, 7.5% DBC
        """
        universe_df = df.filter(regex=r'(SPY|TLT|IEF|GLD|DBC)')
        returns_df = returns(universe_df)
        weights_ls = [.3,.4,.15,.075,.075]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if (1-sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns,data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def us_6040(df):
        """
            60% SPY & 40% IEF
        """
        universe_df = df.filter(regex=r'(SPY|IEF)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.4]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

class TacticalAllocation:

    pass

class PortfolioStrategies:

    log_obj = Logs()

    def __init__(self,object,df):
        self.df = df
        methods_ls = inspect.getmembers(object,predicate=inspect.isfunction)
        self.strategy_name = [method[-1] for method in methods_ls]

    def equity_curves_aggregate(self):

        equity_curves_dd = dict()

        def add_equity_curve(strategy,df):
            equity_curves_dd[strategy.__name__] = strategy(df)
            return equity_curves_dd

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for strategy in self.strategy_name:
                results.append(executor.submit(add_equity_curve,strategy,self.df))

            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[COMPUTATION]: {f.__name__} equity curve is being computed @ \
                            {PortfolioStrategies.log_obj.now_date()}.\n'
                    PortfolioStrategies.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        equity_curves_df = pd.concat(equity_curves_dd,axis=1).droplevel(1,1)
        return equity_curves_df

    @staticmethod
    def insertion(df,object,table):
        """
            Insertion of strategies catalog to database
        """
        df.to_sql(name=table,con=data_obj.conn_obj,if_exists='replace')
        if table in['buy_and_hold','tactical_allocation']:
            print(f'[INSERTION]: {type(object).__name__} strategies catalog has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}')

        if 'performance' in table:
            print(f'[INSERTION]: {type(object).__name__} performance has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}')

# class Backtest:
#
#     @staticmethod
#     def performance(df):


if __name__ == '__main__':

    df = data_obj.simulation(data_obj.universe_ls)
    buy_and_hold_obj = BuyAndHold()
    portfolio_strat_obj = PortfolioStrategies(buy_and_hold_obj,df)
    equity_curves_df = portfolio_strat_obj.equity_curves_aggregate()
    equity_curves_df['average'] = equity_curves_df.sum(axis=1)/equity_curves_df.shape[1]
    portfolio_strat_obj.insertion(equity_curves_df,buy_and_hold_obj)


