import pandas as pd
import inspect
from DataStore import Data
import concurrent.futures
from Logger import Logs
import numpy as np
from datetime import datetime
from Backtest import Table
from scipy.optimize import minimize
from itertools import chain
import pickle
import os
from pathlib import Path
from Logger import Logs

data_obj = Data()
log_obj = Logs()

def returns(df,annualised=False):
    if annualised:
        return ((np.log(df/df.shift())+1)**12) - 1
    else:
        return np.log(df/df.shift())

def weights(weights_ls,df):
    weights_ls = [weights_ls * df.shape[0]]
    weights_ls = np.reshape(np.array(weights_ls), newshape=(df.shape[0],df.shape[1]))
    return weights_ls

def momentum_score(df):
    df_copy = df.apply(lambda x:12*(x/x.shift())+4*(x/x.shift(3))+2*(x/x.shift(6))+(x/x.shift(12)))
    df_copy = df_copy - 19
    return df_copy

def momentum_score_sma(df):
    df_copy = df.apply(lambda x: (x/x.shift(12))-1)
    return df_copy

def equity_curve(returns_df,weights_df):
    weights_df = weights_df.shift()
    equity_curve_df = pd.DataFrame((weights_df.mul(returns_df).sum(axis=1)+1).cumprod())
    return equity_curve_df
"""
    The following strategies were coded up according to @WifeyAlpha instructions.
    However, cash instrument has been replaced by NEAR instead of BIL (BIL was used as cash money market
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
        universe_df = df.filter(regex=r'(SHY$|TLT$|VTI$|IWN$|GLD$)')
        returns_df = returns(universe_df)
        weights_ls = [.2] * returns_df.shape[1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df,weights_df)
        return equity_curve_df

    @staticmethod
    def rob_armott(df):
        """
            20% BNDX,20% LQD,10% VEU,10% VNQ,
            10% SPY,10% TLT,10% TIP,10% DBC
        """
        universe_df = df.filter(regex=r'(BNDX$|LQD$|VEU$|VNQ$|SPY$|TLT$|TIP$|DBC$)')
        returns_df = returns(universe_df)
        weights_ls = [.2,.2,.1,.1,.1,.1,.1,.1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def global_asset_allocation(df):
        """
            18% SPY,13.5% EFA,4.5% EEM, 19.8% LQD, 14.4% BNDX,
            13.5% TLT, 1.8% TIP, 5% DBC, 5% GLD, 4.5% VNQ
        """
        universe_df = df.filter(regex=r'(SPY$|^EFA$|EEM$|LQD$|BNDX$|TLT$|TIP$|DBC$|GLD$|VNQ$)')
        returns_df = returns(universe_df)
        weights_ls = [.18,.135,.045,.198,.144,.135,.018,.05,.05,.045]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def permanent(df):
        """
            25% BIL,25% GLD,25% TLT and 25% SPY
        """
        universe_df = df.filter(regex=r'(BIL$|GLD$|TLT$|SPY$)')
        returns_df = returns(universe_df)
        weights_ls = [.25] * returns_df.shape[1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def desert(df):
        """
            60% IEF, 30% VTI, 10% GLD
        """
        universe_df = df.filter(regex=r'(IEF$|VTI$|GLD$)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.3,.1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def larry(df):
        """
            30% in equities  (15% IWN, 7.5% IWN, 7.5% EEM)
            70% in bonds (IEF)
        """
        universe_df = df.filter(regex=r'(IWN$|IWN$|EEM$|LQD$|IEF$)')
        returns_df = returns(universe_df)
        weights_ls = [.15,.075,.075,.7]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def big_rocks(df):
        """
            60% AGG, 6% SPY, 6% IWD, 6% IWM, 6% IWN, 4% EFV, 4% VNQ, 2% EFA, 2% SCZ, 2% DLS, 2% EEM
        """
        universe_df = df.filter(regex=r'(AGG$|SPY$|IWD$|IWM$|IWN$|EFV$|VNQ$|^EFA$|SCZ$|DLS$|EEM$)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.06,.06,.06,.06,.04,.04,.02,.02,.02,.02]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def sandwich(df):
        """
            50% in equities (20% SPY, 10% SCZ, 8% IWM, 6% EEM, 6% EFA)
            41% in bonds (41% IEF)
            9% in cash and REITs (5% VNQ, 4% NEAR)
        """
        universe_df = df.filter(regex=r'(SPY$|SCZ$|IWM$|EEM$|^EFA$|IEF$|VNQ$|NEAR$)')
        returns_df = returns(universe_df)
        weights_ls = [.2,.1,.08,.06,.06,.41,.05,.04]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def balanced_tax_aware(df):
        """
            38% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM
        """
        universe_df = df.filter(regex=r'(AGG$|SPY$|BIL$|^EFA$|IWM$|VNQ$|DBC$|EEM$)')
        returns_df = returns(universe_df)
        weights_ls = [.38,.15,.15,.13,.05,.05,.05,.04]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def balanced(df):
        """
            33% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM, 2% TIP, 2% BNDX, 1% HYG
        """
        universe_df = df.filter(regex=r'(AGG$|SPY$|BIL$|^EFA$|IWM$|VNQ$|DBC$|EEM$|TIP$|BNDX$|HYG$)')
        returns_df = returns(universe_df)
        weights_ls = [.33,.15,.15,.13,.05,.05,.05,.04,.02,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def income_growth(df):
        """
            37% AGG, 20% BIL, 10% TIP, 9% SPY, 8% EFA, 5% VNQ, 4% HYG, 4% BNDX, 2% IWM, 1% DBC
        """
        universe_df = df.filter(regex=r'(AGG$|BIL$|TIP$|SPY$|^EFA$|VNQ$|HYG$|BNDX$|IWN$|DBC$)')
        returns_df = returns(universe_df)
        weights_ls = [.37,.2,.1,.09,.08,.05,.04,.04,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def income_growth_tax(df):
        """
            55% AGG, 20% BIL, 9% SPY, 8% EFA, 5% VNQ, 2% IWM, 1% DBC
        """
        universe_df = df.filter(regex=r'(AGG$|BIL$|SPY$|^EFA$|VNQ$|IWM$|DBC$)')
        returns_df = returns(universe_df)
        weights_ls = [.55,.2,.09,.08,.05,.02,.01]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def conservative_income(df):
        """
            70% in bonds (40% AGG, 18% BIL, 7% HYG, 5% BNDX),
            25% in cash (25% NEAR), and
            5% in REITs (5% VNQ).
        """
        universe_df = df.filter(regex=r'(AGG$|BIL$|HYG$|BNDX$|NEAR$|VNQ$)')
        returns_df = returns(universe_df)
        weights_ls = [.4,.18,.07,.05,.25,.05]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def conservative_income_tax(df):
        """
            70% in bonds (70% AGG),
            25% in cash (25% NEAR), and
            5% in REITs (5% VNQ).
        """
        universe_df = df.filter(regex=r'(AGG$|NEAR$|VNQ$)')
        returns_df = returns(universe_df)
        weights_ls = [.7,.25,.05]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index,columns=universe_df.columns,data=weights_ls).shift()
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def all_weather(df):
        """
            30% SPY, 40% TLT, 15% IEF, 7.5% GLD, 7.5% DBC
        """
        universe_df = df.filter(regex=r'(^SPY$|^TLT$|^IEF$|^GLD$|^DBC$)')
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
        universe_df = df.filter(regex=r'(SPY$|IEF$)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.4]
        assert sum(weights_ls) == 1
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

class TacticalAllocation:


    @staticmethod
    def ivy(df):
        """
            20% VTI,20% VEU,20% VNQ,20% AGG and 20% DBC + NEAR
        """
        risky_assets_ls = ['VTI','VEU','VNQ','AGG','DBC']
        universe_df = df.filter(regex=r'(VTI|VEU|VNQ|AGG|DBC|NEAR)')
        returns_df = returns(universe_df)
        universe_df = universe_df[risky_assets_ls]
        weights_ls = [.2] * (returns_df.shape[1]-1)
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df[risky_assets_ls])
        weights_df = pd.DataFrame(index=returns_df.index,columns=risky_assets_ls,\
                                  data=weights_ls)
        risky_assets_allocation_df = weights_df.mul((universe_df > universe_df.rolling(window=10).mean()))
        cash_allocation_s = (universe_df>universe_df.rolling(window=10).mean()).apply(lambda x:x.value_counts(False),\
                                                                                      axis=1)[False]*0.2
        cash_allocation_s.name = 'NEAR'
        weights_df = risky_assets_allocation_df.join(cash_allocation_s,how='left')
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def robust_asset_allocation_balanced(df):
        """
            15% VNQ, 20% IEF, 20% DBC, 20% MTUM, 10% IWB, 10% EFA and 10% EFV (Tweet)
            20% VNQ, 10% IEFA, 20% MTUM, 10% IWB, 10% EFA, 10% EFV and 20% IEF (Suggested in the paper)
        """
        risky_assets_ls = ['VNQ','IEFA','MTUM','IWB','EFA','EFV','IEF']
        universe_df = df.filter(regex=r'(VNQ|^IEFA$|MTUM|IWB|^EFA$|EFV|^IEF$|NEAR)')
        risk_free_asset_s = universe_df['NEAR']
        returns_df = returns(universe_df)
        universe_df = universe_df[risky_assets_ls]
        weights_ls = [.2,.1,.2,.1,.1,.1,.2]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, universe_df[risky_assets_ls])
        weights_df = pd.DataFrame(index=returns_df.index, columns=risky_assets_ls,data=weights_ls).shift()
        returns_12M_df = universe_df[risky_assets_ls].pct_change(periods=12)
        risk_free_asset_ret_12M_s = risk_free_asset_s.pct_change(periods=12)
        condition1_df = returns_12M_df.gt(risk_free_asset_ret_12M_s, axis=0).astype(int)
        condition2_df = (universe_df>returns_12M_df).astype(int)
        signal_df = condition1_df.add(condition2_df)
        signal_df = 0.5*signal_df
        risky_assets_allocation_df = weights_df.mul(signal_df)
        cash_allocation_s = 1-risky_assets_allocation_df.sum(axis=1)
        cash_allocation_s.name = 'NEAR'
        weights_df = risky_assets_allocation_df.join(cash_allocation_s, how='left')
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def global_tactical_asset_allocation(df):
        """
            5% IWD, 5% MTUM, 5% IWN, 5% DWAS, 10% EFA, 10% EEM, 5% IEF, 5% BWX, 5% LQD, 5% TLT, 10% DBC,
            10% GLD and 20% VNQ
        """
        risky_assets_ls = ['IWD','MTUM','IWM','DWAS','EFA','EEM','IEF','BWX','LQD','TLT','DBC','GLD','VNQ']
        universe_df = df.filter(regex=r'(IWD|VNQ|MTUM|IWM|DWAS|^EFA$|EEM|^IEF$|BWX|LQD|TLT|DBC|GLD|VNQ|NEAR)')
        returns_df = returns(universe_df)
        universe_df = universe_df[risky_assets_ls]
        weights_ls = [.05,.05,.05,.05,.1,.1,.05,.05,.05,.05,.1,.1,.2]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls, universe_df[risky_assets_ls])
        risky_assets_allocation_df = pd.DataFrame(index=returns_df.index, columns=risky_assets_ls,data=weights_ls)
        price_10M_df = universe_df.rolling(10).mean()
        signal_df = universe_df.gt(price_10M_df)
        risk_free_asset_allocation_s = signal_df.apply(lambda x:x.value_counts(False),axis=1)/signal_df.shape[1]
        risk_free_asset_allocation_s = risk_free_asset_allocation_s[False]
        weights_df = risky_assets_allocation_df.join(risk_free_asset_allocation_s,how='left')
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def diversified_gem_dual_momentum(df):
        """
            43% SPY, 30% AGG, 27% EFA
        """
        universe_df = df.filter(regex=r'(SPY|AGG|^EFA$)')
        returns_df = returns(universe_df)
        weights_ls = [.43,.3,.27]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,universe_df)
        weights_df = pd.DataFrame(index=returns_df.index,columns=universe_df.columns.tolist(),data=weights_ls)
        returns_lookback_dd = dict()
        for i in range(7,13):
            temp_df = universe_df.pct_change(periods=i)
            series_s = temp_df['SPY']>temp_df['AGG']
            returns_lookback_dd[f'{i}_months'] = series_s
        returns_lookback_df = pd.DataFrame(returns_lookback_dd)
        spy_agg_df = returns_lookback_df.apply(lambda x:x.value_counts(False),axis=1)#/returns_lookback_df.shape[1]
        dd = {'SPY':np.where(returns_df.SPY > returns_df.AGG, True, False),\
              'EFA':np.where(returns_df.SPY < returns_df.AGG, True, False)}

        spy_efa_df = pd.DataFrame(data=dd, index=returns_df.index)
        spy_efa_df = spy_efa_df.mul(spy_agg_df[True],axis=0)
        agg_s = spy_agg_df[False]
        agg_s.name = 'AGG'
        spy_agg_efa_df = spy_efa_df.join(agg_s,how='left')
        weights_df = weights_df.mul(spy_agg_efa_df)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def vigilant_asset_allocation_g12(df):
        """
            Risky assets =  SPY, IWM, QQQ, VGK, EWJ, EEM, VNQ, DBC, GLD, TLT, LQD, and HYG
            Safe assets = IEF LQD, and BIL
        """
        risky_assets_ls = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG']
        safe_assets_ls = ['IEF','LQD','BIL']
        regex = [f'^{sym}$' for sym in risky_assets_ls+safe_assets_ls]
        regex = ''.join(('(','|'.join(regex),')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_df = pd.DataFrame(index=returns_df.index,columns=returns_df.columns)
        momentum_score_df = momentum_score(universe_df)
        negative_score_s = (momentum_score_df>0).apply(lambda x:x.value_counts(False),axis=1)[False]
        for date, series_s in momentum_score_df.iterrows():
            n = negative_score_s[date]
            range_n_less_than_4 = list(range(0,4))
            safe_assets_highest_score = series_s[safe_assets_ls].sort_values(ascending=False).index[0]
            risky_assets_highest_score = series_s[risky_assets_ls].sort_values(ascending=False)
            if n>=4:
                weights_df.loc[date,safe_assets_highest_score] = 1.0
            if n in range_n_less_than_4:
                weights_df.loc[date, safe_assets_highest_score] = .25*n
                risky_assets_n3_ls = risky_assets_highest_score[:5].index.tolist()
                weights_df.loc[date,risky_assets_n3_ls] = (1-(.25*n))/5
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def vigilant_asset_allocation_g4(df):
        """
             Risk assets: SPY, EFA, EEM, and AGG
             Safety assets: LQD, IEF, and BIL
        """
        risky_assets_ls = ['SPY','EFA','EEM','AGG']
        safe_assets_ls = ['IEF', 'LQD', 'BIL']
        regex = [f'^{sym}$' for sym in risky_assets_ls + safe_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        momentum_score_df = momentum_score(universe_df)
        risky_assets_negative_score_s = (momentum_score_df[risky_assets_ls] > 0).apply(lambda x: x.value_counts(False),\
                                                                                   axis=1)[False]
        for date, series_s in momentum_score_df.iterrows():
            n = risky_assets_negative_score_s[date]
            safe_assets_highest_score = series_s[safe_assets_ls].sort_values(ascending=False)
            risky_assets_highest_score = series_s[risky_assets_ls].sort_values(ascending=False)
            if n == 0:
                weights_df.loc[date,risky_assets_highest_score.index[0]] = 1
            if n>0:
                weights_df.loc[date,safe_assets_highest_score.index[0]] = 1
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def defensive_asset_allocation(df):
        """
            Risky assets: SPY, IWM, QQQ, VGK, EWJ, EEM, VNQ, DBC, GLD, TLT, HYG, LQD
            Protective assets: SHY, IEF
            Canary assets: EEM, AGG
        """
        risky_assets_ls = ['SPY','IWM','QQQ','VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG','LQD']

        protective_assets_ls = ['SHY', 'IEF', 'BIL']
        canary_assets_ls = ['EEM','AGG']
        regex = [f'^{sym}$' for sym in risky_assets_ls+protective_assets_ls+canary_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        momentum_score_df = momentum_score(universe_df)
        canary_assets_positive_score_s = \
            2*(momentum_score_df[canary_assets_ls] > 0).apply(lambda x: x.value_counts(False),axis=1)[True]
        canary_assets_positive_score_s = canary_assets_positive_score_s.fillna(0)
        for date, series_s in momentum_score_df.iterrows():
            n = canary_assets_positive_score_s[date]
            protective_assets_highest_score = series_s[protective_assets_ls].sort_values(ascending=False).index[0]
            risky_assets_highest_score = series_s[risky_assets_ls].sort_values(ascending=False)
            if n == 2:
                weights_df.loc[date,protective_assets_highest_score] = 1
            if n in [0,1]:
                weights_df.loc[date, protective_assets_highest_score] = .5*n
                weights_df.loc[date, risky_assets_highest_score.index.tolist()] = (1-.5*n)/6
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def gem_dual_momentum(df):
        """
            45% SPY, 28% AGG, 27% EFA and BIL
        """
        risky_assets_ls = ['SPY','AGG','EFA']
        safe_assets_ls = ['BIL']
        regex = [f'^{sym}$' for sym in risky_assets_ls + safe_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_ls = [.45,.28,.27]
        weights_ls = weights(weights_ls, universe_df[risky_assets_ls])
        risky_assets_allocation_df = pd.DataFrame(index=returns_df.index, columns=risky_assets_ls, data=weights_ls)
        spy_efa_bil_returns_12M_df = universe_df[['SPY','EFA','BIL']].apply(lambda x:np.log(x/x.shift(12)))
        spy_bil_s = spy_efa_bil_returns_12M_df.SPY>spy_efa_bil_returns_12M_df.BIL
        spy_efa_s = spy_efa_bil_returns_12M_df.SPY>spy_efa_bil_returns_12M_df.EFA
        signals_dd = {'SPY':(spy_bil_s*spy_efa_s).astype(int),\
                      'EFA':spy_bil_s*(1-spy_efa_s),\
                      'AGG':1-spy_bil_s}
        signals_df = pd.DataFrame(signals_dd)
        weights_df = risky_assets_allocation_df.mul(signals_df)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def quint_switching_filtered(df):
        """
            75% IEF, 10% QQQ, 8% EEM, 4% EFA, 2% TLT, 1% SPY
        """
        risky_assets_ls = ['QQQ','EEM','EFA','TLT','SPY']
        safe_assets_ls = ['IEF']
        regex = [f'^{sym}$' for sym in risky_assets_ls + safe_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_ls = [.75,.1,.08,.04,.02,.1]
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df)
        weights_df = pd.DataFrame(index=returns_df.index,columns=safe_assets_ls+risky_assets_ls,\
                                  data=weights_ls)
        signals_df = pd.DataFrame(index=returns_df.index,columns=returns_df.columns)
        returns_risky_assets_3M_df = universe_df[risky_assets_ls].apply(lambda x:np.log(x/x.shift(3)))
        returns_risky_assets_negative_3M_s = \
            (returns_risky_assets_3M_df>0).apply(lambda x:x.value_counts(),axis=1)[False]
        for date,series_s in returns_risky_assets_3M_df.iterrows():
            n = returns_risky_assets_negative_3M_s[date]
            if n > 0:
                signals_df.loc[date,safe_assets_ls] = 1
            if n == len(risky_assets_ls):
                signals_df.loc[date,series_s.sort_values(ascending=False).index[0]] = 1
        weights_df = weights_df.mul(signals_df)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def composite_dual_momentum(df):
        """
            25% Equities: SPY and EFA
            25% Real Estate: VNQ and REM
            25% Stress: GLD and TLT
            25% Bonds: HYG and LQD
            https://allocatesmartly.com/antonaccis-composite-dual-momentum/

        """
        equities_assets_ls = ['SPY','EFA']
        real_estate_assets_ls = ['VNQ', 'REM']
        stress_assets_ls = ['GLD', 'TLT']
        bonds_assets_ls = ['HYG', 'LQD']
        asset_classes_dd = {'equities':equities_assets_ls,'real_estate':real_estate_assets_ls,\
                            'stress':stress_assets_ls,'bonds':bonds_assets_ls}
        regex = [f'^{sym}$' for sym in \
        equities_assets_ls + real_estate_assets_ls + stress_assets_ls + bonds_assets_ls + ['BIL']]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        returns_assets_12M_df = universe_df.apply(lambda x:np.log(x/x.shift(12)))
        returns_bil_12M_s = returns_assets_12M_df.pop('BIL')
        risky_assets_allocation_df = pd.DataFrame(index=returns_df.index,\
        columns=equities_assets_ls+real_estate_assets_ls+stress_assets_ls+bonds_assets_ls)
        safe_asset_allocation_s = pd.DataFrame(index=returns_df.index,columns=list(asset_classes_dd.keys()))
        returns_df = returns(universe_df)
        for asset_class,sym_ls in asset_classes_dd.items():
            temp_12M_df = returns_assets_12M_df[sym_ls]
            signals_asset_class_df = pd.DataFrame(data={sym_ls[0]:temp_12M_df[sym_ls[0]]>temp_12M_df[sym_ls[1]],\
                                                        sym_ls[1]:temp_12M_df[sym_ls[0]]<temp_12M_df[sym_ls[1]]})
            asset_class_bil_df = universe_df[sym_ls].gt(returns_bil_12M_s,axis=0)
            risky_assets_allocation_df[sym_ls] = .25*signals_asset_class_df.mul(asset_class_bil_df)
            temp_condition_df = asset_class_bil_df.apply(lambda x:x.value_counts(),axis=1).fillna(0)
            safe_asset_allocation_s[asset_class] = (temp_condition_df[False]>temp_condition_df[True]).astype(int)
        safe_asset_allocation_s = .25*safe_asset_allocation_s.sum(axis=1)
        safe_asset_allocation_s.name = 'BIL'
        weights_df = risky_assets_allocation_df.join(safe_asset_allocation_s,how='left')
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def trend_is_our_friend(df):
        """
            20% Equities, 26% Bonds, and 54% Cash, Commodities and Real Estate
        """
        universe_dd = {'equities':['SPY'],\
                       'bonds':['BIL'],\
                       'CCRE':['NEAR','VNQ','DBC']}
        assets_ls = list(universe_dd.items())
        assets_ls = list(chain(*[pair[-1] for pair in assets_ls]))
        regex = [f'^{sym}$' for sym in assets_ls]
        regex = ''.join(('(','|'.join(regex),')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        assets_ls.remove('NEAR')
        query = data_obj.write_query_symbol(symbol='BIL')
        bil_s = data_obj.query(query,set_index=True)['BIL']
        bil_s = np.log(bil_s/bil_s.shift())
        vol_df = universe_df.apply(lambda x:np.log(x/x.shift())).rolling(18).std()
        vol_df = vol_df.drop('NEAR',axis=1)
        vol_df = vol_df.subtract(bil_s,axis=0)
        universe_10M_df = universe_df.rolling(10).mean()
        signals_df = (universe_df>universe_10M_df).astype(int)
        signals_df = signals_df.reset_index()
        signals_df = signals_df.drop('NEAR',axis=1)
        weights_df = pd.DataFrame(index=returns_df.index,columns=assets_ls)
        for idx,series_s in signals_df.iterrows():
            if idx>=18:
                date = series_s.pop('index')
                series_vol_s = vol_df.loc[date, :]
                k = 1/sum(series_vol_s**(-1))
                weights_ls = k*(1/series_vol_s)
                weights_df.loc[date,series_vol_s.index.tolist()] = weights_ls
        signals_df = signals_df.set_index('index')
        weights_df = weights_df.mul(signals_df)
        weights_df['NEAR'] = 1-weights_df.sum(axis=1)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def kipnis_defensive_adaptive_asset_allocation(df):
        """
             Investment Universe: SPY, VGK, EWJ, EEM, VNQ, RWX, IEF, TLT, DBC, and GLD
             Crash Protection: IEF and Cash (NEAR)
             Canary: EEM and AGG
        """
        investment_universe_ls = ['SPY', 'VGK', 'EWJ', 'EEM', 'VNQ', 'RWX', 'TLT', 'DBC', 'GLD']
        crash_protection_ls = ['IEF','NEAR']
        canary_ls = ['EEM','AGG']
        regex = [f'^{sym}$' for sym in investment_universe_ls + crash_protection_ls + canary_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df[investment_universe_ls+crash_protection_ls])
        momentum_score_df = momentum_score(universe_df).reset_index()
        canary_assets_positive_score_s = (momentum_score_df[canary_ls]>0).apply(lambda x:x.value_counts(),axis=1)
        canary_assets_positive_score_s.index = returns_df.index
        canary_assets_positive_score_s = canary_assets_positive_score_s.fillna(0)
        canary_assets_positive_score_s[True] = 2-canary_assets_positive_score_s[False]
        canary_assets_positive_score_s = canary_assets_positive_score_s[True]
        ief_near_momentum_dd = {'IEF':(momentum_score_df[crash_protection_ls].apply(lambda x:x['IEF']>x['NEAR'],\
        axis=1)).astype(int),\
        'NEAR':(momentum_score_df[crash_protection_ls].apply(lambda x:x['IEF']<x['NEAR'],axis=1)).astype(int)}
        ief_near_momentum_df = pd.DataFrame(ief_near_momentum_dd)
        ief_near_momentum_df.index = returns_df.index
        ief_near_momentum_df.loc[:12,'NEAR'] = 1
        weights_df = pd.DataFrame(index=returns_df.index,columns=investment_universe_ls)
        for idx,series_s in momentum_score_df.iterrows():
            if idx>12:
                date = series_s.pop('index')
                factor = 1-.5*canary_assets_positive_score_s[date]
                investment_universe_top_momentum = \
                    (series_s[investment_universe_ls].sort_values(ascending=False) > 0)
                if investment_universe_top_momentum.sum()>=5:
                    top5_investment_universe_ls = investment_universe_top_momentum.index[:5]
                else:
                    top5_investment_universe_ls =\
                        investment_universe_top_momentum.index[:investment_universe_top_momentum.sum()]
                cov_df = \
                    ((12*universe_df.loc[:date,top5_investment_universe_ls.tolist()].pct_change().cov()).add(\
                    3*universe_df.loc[:date,top5_investment_universe_ls.tolist()].pct_change(4).cov())).add(\
                        universe_df.loc[:date,top5_investment_universe_ls.tolist()].pct_change(12).cov())
                cov_df = (1/19)*cov_df
                weights_ls = np.array([1/len(cov_df.index)]*len(cov_df.index))
                res = minimize(fun=lambda x,cov:np.dot(x,np.dot(cov,np.transpose(x))),\
                               x0=weights_ls,args=(cov_df),constraints={'type':'eq',\
                                                                        'fun':lambda x:sum(x)-1})
                weights_ls = list(res.x)
                weights_df.loc[date,top5_investment_universe_ls.tolist()] = weights_ls
                weights_df.loc[date,top5_investment_universe_ls.tolist()] = factor*weights_df.loc[date,\
                                                                        top5_investment_universe_ls.tolist()]
                ief_near_momentum_df.loc[date,:] = factor*ief_near_momentum_df.loc[date,:]
        ief_near_momentum_df.loc[:,'NEAR'] = np.where(weights_df.sum(axis=1)>=ief_near_momentum_df.loc[:,'NEAR'],0,1)
        weights_df = weights_df.join(ief_near_momentum_df, how='left')
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def adaptive_asset_allocation(df):
        """
            IEF,SPY,VNQ,RWX,DBC,EEM,VGK,TLT,GLD,EWJ and NEAR
        """
        universe_assets_ls = ['IEF','SPY','VNQ','RWX','DBC','EEM','VGK','TLT','GLD','EWJ','NEAR']
        risk_assets_ls = universe_assets_ls[::]
        risk_assets_ls.remove('NEAR')
        regex = [f'^{sym}$' for sym in universe_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        weights_df = pd.DataFrame(index=returns_df.index,columns=universe_df.columns)
        risky_assets_returns_6M_df = universe_df[risk_assets_ls].apply(lambda x:np.log(x/x.shift(6)))
        risky_assets_returns_6M_df = risky_assets_returns_6M_df.reset_index()
        for idx, series_s in risky_assets_returns_6M_df.iterrows():
            if idx >= 6:
                date = series_s.pop('index')
                five_assets_ls = series_s.sort_values(ascending=False).index[:5].tolist()
                cov_df = returns_df[five_assets_ls].rolling(2).cov().loc[date, :]
                if weights_df.loc[weights_df.index[idx-1],five_assets_ls].sum() < 5:
                    weights_x0_ls = [1/cov_df.shape[0]] * cov_df.shape[0]
                else:
                    weights_x0_ls = weights_df[five_assets_ls].iloc[idx-1,:]
                res = minimize(lambda x,cov: np.dot(x,np.dot(cov,np.transpose(x))),\
                x0=weights_x0_ls,args=(cov_df),constraints={'type':'eq','fun':lambda x:sum(x)-1})
                weights_ls = list(res.x)
                weights_df.loc[date,five_assets_ls] = weights_ls
        weights_df['NEAR'] = 1 - weights_df.sum(axis=1)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def protective_asset_allocation(df):
        """
            51% IEF, 6% IWM, 6% QQQ, 5% VNQ, 5% SPY, 5% VGK, 5% EEM, 4% EWJ, 3% DBC,\
            3% TLT, 3% GLD, 2% HYG and 2% LQD
        """
        risky_assets_ls = ['IWM','QQQ','VNQ','SPY','VGK','EEM','EWJ','DBC','TLT','GLD','HYG','LQD']
        safe_assets_ls = ['IEF']
        weights_ls = [.51,.06,.06,.05,.05,.05,.05,.04,.03,.03,.03,.02,.02]
        regex = [f'^{sym}$' for sym in safe_assets_ls+risky_assets_ls]
        regex = ''.join(('(', '|'.join(regex), ')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        try:
            assert sum(weights_ls) == 1
        except AssertionError:
            if abs(1 - sum(weights_ls)) < 1e-8:
                pass
        weights_ls = weights(weights_ls,returns_df[safe_assets_ls+risky_assets_ls])
        weights_df = pd.DataFrame(index=returns_df.index, columns=safe_assets_ls+risky_assets_ls, \
                                  data=weights_ls).shift()
        signals_df = pd.DataFrame(index=returns_df.index,columns=safe_assets_ls+risky_assets_ls)
        momentum_score_sma_df = momentum_score_sma(universe_df[risky_assets_ls])
        positive_n_score_s = (momentum_score_sma_df>0).apply(lambda x:x.value_counts(False),axis=1)[True]
        for date,series_s in momentum_score_sma_df.iterrows():
            risky_assets_highest_score = series_s[risky_assets_ls].sort_values(ascending=False)
            n = positive_n_score_s[date]
            if n <= 6:
                signals_df.loc[date,safe_assets_ls] = 1
            if n >= 7:
                safe_assets_portion = (12-n)/6
                signals_df.loc[date,safe_assets_ls] = safe_assets_portion
                signals_df.loc[date,risky_assets_highest_score.index.tolist()] = (1-((12-n)/6))/6
        weights_df = weights_df.mul(signals_df)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def generalised_protective_momentum(df):
        """
            Risk:  SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, DBC, GLD, HYG, and LQD
            Safety: BIL, IEF
        """
        risky_assets_ls = ['SPY','QQQ','IWM','VGK','EWJ','EEM','VNQ','DBC','GLD','HYG','LQD']
        safety_assets_ls = ['BIL','IEF']
        regex = [f'^{sym}$' for sym in risky_assets_ls+safety_assets_ls]
        regex = ''.join(('(','|'.join(regex),')'))
        universe_df = df.filter(regex=f'{regex}')
        returns_df = returns(universe_df)
        returns_df['weighted_risk_assets'] = returns_df[risky_assets_ls].sum(axis=1)
        ri_df = (universe_df.pct_change(periods=1)+universe_df.pct_change(periods=3)\
                 +universe_df.pct_change(periods=6)+universe_df.pct_change(periods=12))
        ci_df = returns_df.rolling(12).cov()
        ci_weighted_risk_assets_df = ci_df.loc[ci_df.index.get_level_values(1)=='weighted_risk_assets',:]
        weights_df = pd.DataFrame(index=universe_df.index,columns=universe_df.columns)
        ci_weighted_risk_assets_df.index = ci_weighted_risk_assets_df.index.droplevel(1)
        ci_weighted_risk_assets_df = ci_weighted_risk_assets_df.drop('weighted_risk_assets',axis=1)
        n_positive_score_s = (((1-ci_weighted_risk_assets_df)*ri_df)>0).sum(axis=1)
        for date,series_s in ci_weighted_risk_assets_df.iterrows():
            n = n_positive_score_s[date]
            if n<=6:
                highest_score_safe_asset = series_s[safety_assets_ls].sort_values(ascending=False).index[0]
                weights_df.loc[date,highest_score_safe_asset] = 1
            else:
                top3_highest_score_risky_assets_ls = series_s[risky_assets_ls].sort_values(ascending=False).index[:3]
                weights_df.loc[date, highest_score_safe_asset] = (12-n)/6
                remaining_portion = (12-n)/6
                weights_df.loc[date,top3_highest_score_risky_assets_ls] = remaining_portion/3
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

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
        equity_curves_df = pd.concat(equity_curves_dd,axis=1).droplevel(1,1)
        return equity_curves_df

    @staticmethod
    def insertion(df,object,table):
        """
            Insertion of strategies catalog to database
        """
        df.to_sql(name=table,con=data_obj.conn_obj,if_exists='replace')
        if table in['buy_and_hold','tactical_allocation']:
            msg = f'[INSERTION]: {type(object).__name__} strategies catalog has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}'


        if 'performance' in table:
            msg = f'[INSERTION]: {type(object).__name__} performance has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}'

        if 'returns' in table:
            msg = f'[INSERTION]: {type(object).__name__} returns has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}'
        PortfolioStrategies.log_obj.log_msg(msg)

    @staticmethod
    def to_pickle(dd,object):
        path = os.path.abspath(path=Path('./DataStore'))
        try:
            pickle_out = open(f'{path}/{type(object).__name__}.pickle','wb')
        except FileNotFoundError:
            path = os.path.abspath(path=Path('../DataStore'))
            pickle_out = open(f'{path}/{type(object).__name__}.pickle', 'wb')
        pickle.dump(dd,pickle_out)
        pickle_out.close()
        msg = f'[INSERTION]: {type(object).__name__} rolling performance has been created @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}'
        PortfolioStrategies.log_obj.log_msg(msg)



if __name__ == '__main__':

    def update(allocations=['buy_and_hold','tactical_allocation']):
        query = data_obj.write_query_price()
        df = data_obj.query(query, set_index=True)
        for allocation in allocations:
            if allocation == 'buy_and_hold':
                allocation_obj = BuyAndHold()
            if allocation == 'tactical_allocation':
                allocation_obj = TacticalAllocation()
            table_returns = '_'.join((allocation, 'returns'))
            table_performance = '_'.join((allocation,'performance'))
            portfolio_strat_obj = PortfolioStrategies(allocation_obj,df)
            equity_curves_df = portfolio_strat_obj.equity_curves_aggregate()
            equity_curves_df['average'] = equity_curves_df.sum(axis=1)/equity_curves_df.shape[1]
            portfolio_strat_obj.insertion(equity_curves_df,allocation_obj,table=allocation)
            portfolio_strat_obj.insertion(equity_curves_df-1,allocation_obj,table=table_returns)
            query = data_obj.write_query_returns(allocation=allocation)
            returns_df = data_obj.query(query,set_index=True)
            returns_df.index.name = 'time'
            perf_obj = Table(returns_df)
            perf_df = perf_obj.table_aggregate()
            portfolio_strat_obj.insertion(perf_df,allocation_obj,table=table_performance)
            rolling_perf_dd = perf_obj.rolling_aggregate()
            portfolio_strat_obj.to_pickle(rolling_perf_dd,allocation_obj)

    update()



