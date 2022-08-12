import pandas as pd
import inspect
from DataStore import Data
import concurrent.futures
import pdb
from Logger import Logs
import numpy as np
from datetime import datetime
from Backtest import Table
from Backtest import Performance
import pickle
import os


data_obj = Data()

def returns(df,annualised=False):
    if annualised:
        return ((np.log(df/df.shift())+1)**12) - 1
    else:
        return np.log(df/df.shift())

def weights(weights_ls,df):
    weights_ls = [weights_ls * df.shape[0]]#[weights_ls * (df.shape[0]+df.shape[1])]
    weights_ls = np.reshape(np.array(weights_ls), newshape=(df.shape[0],df.shape[1]))
    return weights_ls

def momentum_score(df):
    df_copy = df.apply(lambda x:12*(x/x.shift())+4*(x/x.shift(3))+2*(x/x.shift(6))+(x/x.shift(12)))
    df_copy = df_copy - 19
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
        universe_df = df.filter(regex=r'(SPY$|$TLT$|IEF$|GLD$|DBC$)')
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
                                  data=weights_ls).shift()
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
        risky_assets_allocation_df = pd.DataFrame(index=returns_df.index, columns=risky_assets_ls,data=weights_ls).shift()
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
        spy_agg_df = returns_lookback_df.apply(lambda x:x.value_counts(True),axis=1)#/returns_lookback_df.shape[1]
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



    # @staticmethod
    # def generalised_protective_momentum(df):
    #     """
    #         Risk:  SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, DBC, GLD, HYG, and LQD
    #         Safety: BIL, IEF
    #     """
    #     risky_assets_ls = ['SPY','QQQ','IWM','VGK','EWJ','EEM','VNQ','DBC','GLD','HYG','LQD']
    #     safety_assets_ls = ['BIL','IEF']
    #     regex = [f'^{sym}$' for sym in risky_assets_ls+safety_assets_ls]
    #     regex = ''.join(('(','|'.join(regex),')'))
    #     universe_df = df.filter(regex=f'{regex}')
    #     returns_df = returns(universe_df)
    #     ri_df = (universe_df.pct_change(periods=1)+universe_df.pct_change(periods=3)\
    #              +universe_df.pct_change(periods=6)+universe_df.pct_change(periods=12))#.apply(lambda x:x.mean())
    #     pdb.set_trace()






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
                except Exception:
                    # msg = f'[COMPUTATION]: ERROR in {f.exception_info} equity curve computation @ \
                    #                             {PortfolioStrategies.log_obj.now_date()}.\n'
                    # print(msg)
                    # PortfolioStrategies.log_obj.log_msg(msg)
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

        if 'returns' in table:
            print(
                f'[INSERTION]: {type(object).__name__} returns has been inserted into the database @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}')

    @staticmethod
    def to_pickle(dd,object):
        path = os.path.abspath('/Users/emmanueldjanga/wifeyAlpha/DataStore')
        pickle_out = open(f'{path}/{type(object).__name__}.pickle','wb')
        pickle.dump(dd,pickle_out)
        pickle_out.close()
        print(f'[INSERTION]: {type(object).__name__} rolling performance has been created @ {datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")}')



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
    update()#['tactical_allocation']
    #query = data_obj.write_query_returns(allocation='buy_and_hold')
    #df = data_obj.query(query,set_index=True)
    #port_obj.equity_curves_aggregate()
    #pdb.set_trace()
    #query = data_obj.write_query_price()
    #df = data_obj.query(query, set_index=True)
    #allocation_obj = TacticalAllocation()
    #allocation_obj.vigilant_asset_allocation_g4(df)
    #port_obj = PortfolioStrategies(allocation_obj,df)
    #port_obj.equity_curves_aggregate()



