import pandas as pd
from src.tradingDashboard.StrategiesBase import BuyAndHold
from src.tradingDashboard.StrategiesBase import TacticalAssetAllocation
from src.tradingDashboard.StrategiesBase import is_end_business_month
from src.tradingDashboard.data import Data
from src.tradingDashboard.StrategiesBase import momentum_score
from src.tradingDashboard.StrategiesBase import momentum_score_sma
from src.tradingDashboard.StrategiesBase import vol_estimate
from src.tradingDashboard.StrategiesBase import cov_matrix
from src.tradingDashboard.StrategiesBase import ri
from src.tradingDashboard.StrategiesBase import ci
import datetime
import numpy as np
import pdb
#import vectorbt as vbt
from scipy.optimize import minimize
"""
    Buy and Hold strategies
"""
class GoldenButterflyStrategy(BuyAndHold):

    def __init__(self, instruments=['SHY', 'TLT', 'VTI', 'IWN', 'GLD'], name='gold_butterfly'):
        holdings = [.2 for _ in range(0, len(instruments))]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class RobArmottStrategy(BuyAndHold):
    """
        20% BNDX,20% LQD,10% VEU,10% VNQ,
        10% SPY,10% TLT,10% TIP,10% DBC
    """

    def __init__(self, instruments=['BNDX', 'LQD', 'VEU', 'VNQ', 'SPY', 'TLT', 'TIP', 'DBC'],
                 name='rob_armott'):
        holdings = [.2, .2, .1, .1, .1, .1, .1, .1]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class GlobalAssetAllocationStrategy(BuyAndHold):
    """
        18% SPY,13.5% EFA,4.5% EEM, 19.8% LQD, 14.4% BNDX,
        13.5% TLT, 1.8% TIP, 5% DBC, 5% GLD, 4.5% VNQ
    """

    def __init__(self, instruments=['SPY', 'EFA', 'EEM', 'LQD', 'BNDX', 'TLT', 'TIP', 'DBC', 'GLD', 'VNQ'],
                 name='global_asset_allocation'):
        holdings = [.18, .135, .045, .198, .144, .135, .018, .05, .05, .045]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class PermanentStrategy(BuyAndHold):
    """
        25% BIL,25% GLD,25% TLT and 25% SPY
    """

    def __init__(self, instruments=['BIL', 'GLD', 'TLT', 'SPY'], name='permanent'):
        holdings = [.25 for _ in range(0, len(instruments))]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class DesertStrategy(BuyAndHold):
    """
        60% IEF, 30% VTI, 10% GLD
    """

    def __init__(self, instruments=['IEF', 'VTI', 'GLD'], name='desert'):
        holdings = [.6, .3, .1]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class LarryStrategy(BuyAndHold):
    """
        30% in equities  (15% IWN, 7.5% IWN, 7.5% EEM)
        70% in bonds (IEF)
    """

    def __init__(self, instruments=['IWN', 'DLS', 'EEM', 'IEF'], name='larry'):
        holdings = [.15, .075, .075, .7]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class BigRocksStrategy(BuyAndHold):
    """
        60% AGG, 6% SPY, 6% IWD, 6% IWM, 6% IWN, 4% EFV, 4% VNQ, 2% EFA, 2% SCZ, 2% DLS, 2% EEM
    """

    def __init__(self, instruments=['AGG', 'SPY', 'IWD', 'IWM', 'IWN', 'EFV', 'VNQ', 'EFA', 'SCZ', 'DLS', 'EEM'],
                 name='big_rocks'):
        holdings = [.6, .06, .06, .06, .06, .04, .04, .02, .02, .02, .02]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class SandwichStrategy(BuyAndHold):
    """
        50% in equities (20% SPY, 10% SCZ, 8% IWM, 6% EEM, 6% EFA)
        41% in bonds (41% IEF)
        9% in cash and REITs (5% VNQ, 4% NEAR)
    """

    def __init__(self, instruments=['SPY', 'SCZ', 'IWM', 'EEM', 'EFA', 'IEF', 'VNQ', 'NEAR'],
                 name='sandwich'):
        holdings = [.2, .1, .08, .06, .06, .41, .05, .04]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class BalancedTaxAwareStrategy(BuyAndHold):
    """
        38% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM
    """

    def __init__(self, instruments=['AGG', 'SPY', 'BIL', 'EFA', 'IWM', 'VNQ', 'DBC', 'EEM'],
                 name='balanced_tax_aware'):
        holdings = [.38, .15, .15, .13, .05, .05, .05, .04]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class BalancedStrategy(BuyAndHold):
    """
        33% AGG, 15% SPY, 15% BIL, 13% EFA, 5% IWM, 5% VNQ, 5% DBC, 4% EEM, 2% TIP, 2% BNDX, 1% HYG
    """

    def __init__(self, instruments=['AGG', 'SPY', 'BIL', 'EFA', 'IWM', 'VNQ', 'DBC', 'EEM', 'TIP', 'BNDX', 'HYG'],
                 name='balanced'):
        holdings = [.33, .15, .15, .13, .05, .05, .05, .04, .92, .02, .01]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class IncomeGrowthStrategy(BuyAndHold):
    """
        37% AGG, 20% BIL, 10% TIP, 9% SPY, 8% EFA, 5% VNQ, 4% HYG, 4% BNDX, 2% IWM, 1% DBC
    """

    def __init__(self, instruments=['AGG', 'BIL', 'TIP', 'SPY', 'EFA', 'VNQ', 'HYG', 'BNDX', 'IWM', 'DBC'],
                 name='income_growth'):
        holdings = [.37, .2, .1, .09, .08, .05, .04, .04, .02, .01]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class IncomeGrowthTaxStrategy(BuyAndHold):
    """
        55% AGG, 20% BIL, 9% SPY, 8% EFA, 5% VNQ, 2% IWM, 1% DBC
    """

    def __init__(self, instruments=['AGG', 'BIL', 'SPY', 'EFA', 'VNQ', 'IWM', 'DBC'],
                 name='income_growth_tax'):
        holdings = [.55, .2, .09, .08, .05, .02, .01]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class ConservativeIncomeStrategy(BuyAndHold):
    """
        70% in bonds (40% AGG, 18% BIL, 7% HYG, 5% BNDX),
        25% in cash (25% NEAR), and
        5% in REITs (5% VNQ).
    """

    def __init__(self, instruments=['AGG', 'BIL', 'HYG', 'BNDX', 'NEAR', 'VNQ'],
                 name='conservative_income'):
        holdings = [.4, .18, .07, .05, .25, .05]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class ConservativeIncomeTaxStrategy(BuyAndHold):
    """
        70% in bonds (70% AGG),
        25% in cash (25% NEAR), and
        5% in REITs (5% VNQ).
    """

    def __init__(self, instruments=['AGG', 'NEAR', 'VNQ'],
                 name='conservative_income_tax'):
        holdings = [.7, .25, .05]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class AllWeatherStrategy(BuyAndHold):
    """
        30% SPY, 40% TLT, 15% IEF, 7.5% GLD, 7.5% DBC
    """

    def __init__(self, instruments=['SPY', 'TLT', 'IEF', 'GLD', 'DBC'],
                 name='all_weather'):
        holdings = [.3, .4, .15, .075, .075]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
class US6040Strategy(BuyAndHold):
    """
        60% SPY & 40% IEF
    """
    def __init__(self, instruments=['SPY', 'IEF'],
                 name='us_6040'):
        holdings = [.6, .4]
        super().__init__(instruments=instruments, name=name, holdings=holdings)
"""
    Tactical Asset Allocation strategies
"""
class IvyStrategy(TacticalAssetAllocation):
        """
            VTI,VEU,VNQ,AGG,DBC
        """
        def __init__(self,instruments=['VTI','VEU','VNQ','AGG','DBC'],name='ivy'):
            super().__init__(name=name,instruments=instruments)
        def rule(self,price):
            price_copy = price.loc[:,self.instruments].copy()
            universe_monthly_df = price_copy.resample('BM').last()
            universe_monthly_df = universe_monthly_df.iloc[:-1, :]
            universe_10M_rolling_mean_df = (universe_monthly_df > universe_monthly_df.rolling(window=10).mean()).astype(
                bool)
            signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
            for date, series_s in price_copy.iterrows():
                if is_end_business_month(date):
                    instrument_s = universe_10M_rolling_mean_df.loc[date, :][
                        universe_10M_rolling_mean_df.loc[date, :] == True]
                    if instrument_s.any():
                        signals.loc[date, instrument_s[instrument_s == True].index.tolist()] = 1
            signals.replace(np.nan, 0, inplace=True)
            signals = signals.values
            return signals
class RobustAssetAllocationBalancedStrategy(TacticalAssetAllocation):
    """
        15% VNQ, 20% IEF, 20% DBC, 20% MTUM, 10% IWB, 10% EFA and 10% EFV (Tweet)
        20% VNQ, 10% IEFA, 20% MTUM, 10% IWB, 10% EFA, 10% EFV and 20% IEF (Suggested in the paper)
    """

    def __init__(self,
                 instruments=['VNQ','IEFA','MTUM','IWB','EFA','EFV','IEF','BIL'],
                 name='robust_asset_allocation_balanced'):
        super().__init__(name=name,instruments=instruments)
        self.holdings = pd.Series(index=self.instruments[:-1],data=[.2,.1,.2,.1,.1,.1,.2])
    def rule(self,price):
        price_copy = price.loc[:,self.instruments].copy()
        returns_12M_df = price_copy.div(price_copy.shift(252)).apply(lambda x:np.log(x))
        returns_12M_BIL_s = returns_12M_df.pop('BIL')
        signals_returns_12M_df = (returns_12M_df.gt(returns_12M_BIL_s,axis=0)).astype(int)
        price_copy.drop('BIL',axis=1,inplace=True)
        signals_ma_12M_df = (price_copy>price_copy.rolling(window=252).mean()).astype(int)
        combination_of_two_signals_df = signals_ma_12M_df+signals_returns_12M_df
        signals = pd.DataFrame(index=price_copy.index,columns=price_copy.columns)
        for date,series_s in combination_of_two_signals_df.iterrows():
            if is_end_business_month(date):
                signals.loc[date,series_s.index] = .5*series_s*self.holdings
        signals['BIL'] = 0
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class DiversifiedGEMDualMomentumStrategy(TacticalAssetAllocation):
    """
        43% SPY, 30% AGG, 27% EFA
    """
    def __init__(self,
                 instruments=['SPY','AGG','EFA'],
                 name='diversified_gem_dual_momentum'):
        super().__init__(name=name,instruments=instruments)
        self.holdings = pd.Series(index=self.instruments,data=[.43,.3,.27])
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        month = int(252/12)
        returns_lookback_dd = dict()
        for i in range(7,13):
            temp_df = price_copy.div(price_copy.shift(int(i*month))).apply(lambda x:np.log(x))
            series_s = temp_df['SPY']>temp_df['AGG']
            returns_lookback_dd[f'{i}_months'] = series_s
        # Fist part: comparison SPY and AGG
        spy_agg_returns_lookback_df = pd.DataFrame(returns_lookback_dd)
        spy_agg_df = spy_agg_returns_lookback_df.apply(lambda x: x.value_counts(False), axis=1)
        # Second part: comparison SPY and EFA
        returns_lookback_dd = dict()
        for i in range(7,13):
            temp_df = price_copy.div(price_copy.shift(int(i*month))).apply(lambda x:np.log(x))
            series_s = temp_df['SPY']>temp_df['AGG']
            returns_lookback_dd[f'{i}_months'] = series_s
        spy_efa_returns_lookback_dd = pd.DataFrame(returns_lookback_dd)
        spy_efa_returns_lookback_df = pd.DataFrame(spy_efa_returns_lookback_dd)
        spy_efa_df = spy_efa_returns_lookback_df.apply(lambda x: x.value_counts(False), axis=1)
        signals = pd.DataFrame(index=price_copy.index,columns=price_copy.columns,data=np.nan)
        for date,spy_gt_agg_s in spy_agg_df.iterrows():
            if is_end_business_month(date):
                """
                    If at least one np.nan, one cannot apply strategy's logic --> cash 
                """
                if spy_gt_agg_s.isna().sum()==0:
                    if spy_gt_agg_s[False]>spy_gt_agg_s[True]:
                        signals.loc[date,'AGG'] = True
                    else:
                        spy_gt_efa_s = spy_efa_df.loc[date,:]
                        if spy_gt_efa_s[True]>spy_gt_efa_s[False]:
                            signals.loc[date,'SPY'] = True
                        else:
                            signals.loc[date,'EFA'] = True
        signals.replace(np.nan,False,inplace=True)
        signals = signals.mul(self.holdings)
        return signals.values
class VigilantAssetAllocationG12Strategy(TacticalAssetAllocation):
    """
        Risky assets = SPY, IWM, QQQ, VGK, EWJ, EEM, VNQ, DBC, GLD, TLT, LQD, and HYG
        Safe assets = IEF and BIL
    """
    def __init__(self,
                 instruments=['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG',
                              'IEF','BIL'],
                 risky = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG'],
                 safe = ['IEF','BIL'],
                 name='vigilant_asset_allocation_g12'):
        super().__init__(name=name,instruments=instruments)
        self.risky = risky
        self.safe = safe
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        close_price = price_copy.resample('BM').last().resample('B').last().ffill().loc[:datetime.datetime.today(),:]
        momentum_score_df = momentum_score(price_copy,close_price).ffill()
        momentum_score_df = momentum_score_df.loc[price_copy.index, :]
        signals = pd.DataFrame(index=price_copy.index,columns=price_copy.columns,data=np.nan)
        for date,series_s in momentum_score_df.iterrows():
            if is_end_business_month(date):
                risky_s = series_s[self.risky]
                safe_s = series_s[self.safe]
                if risky_s.isna().sum() < price_copy.shape[1]-len(self.safe):
                    negative_score_s = risky_s[risky_s<0]
                    """
                        Depending on length of negative_score_s allocate resources accordingly
                    """
                    safe_asset_factor = (1-min(4,len(negative_score_s))*.25)
                    if safe_s.isna().sum() < safe_s.shape[0]:
                        top_safe = safe_s.sort_values(ascending=False).index[0]
                        signals.loc[date,top_safe] = safe_asset_factor
                        risky_s = risky_s.sort_values(ascending=False)[:5].dropna()
                        signals.loc[date,risky_s.index.tolist()] = (1-safe_asset_factor)/risky_s.shape[0]
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class VigilantAssetAllocationG4Strategy(TacticalAssetAllocation):
    """
         Risk assets: SPY, EFA, EEM, LQD, and AGG
         Safety assets: IEF and BIL
    """
    def __init__(self,
                 instruments=['SPY','EFA','EEM','LQD','AGG','IEF','BIL'],
                 risky=['SPY','EFA','EEM','LQD','AGG'],
                 safe=['IEF', 'BIL'],
                 name='vigilant_asset_allocation_g4'):
        super().__init__(name=name, instruments=instruments)
        self.risky = risky
        self.safe = safe
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        close_price = price_copy.resample('BM').last().resample('B').last().ffill().loc[:datetime.datetime.today(), :]
        momentum_score_df = momentum_score(price_copy, close_price).ffill()
        momentum_score_df = momentum_score_df.loc[price_copy.index, :]
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date,series_s in momentum_score_df.iterrows():
            if is_end_business_month(date):
                risky_s = series_s[self.risky].dropna()
                if risky_s.isna().sum()<risky_s.shape[0]:
                    safe_s = series_s[self.safe]
                    if risky_s[risky_s<0].shape[0] > 0:
                        if safe_s.isna().sum()<len(self.safe):
                            signals.loc[date,safe_s.sort_values(ascending=False).index[0]] = 1
                    else:
                        signals.loc[date,risky_s.sort_values(ascending=False).index[0]] = 1
        signals.replace(np.nan,0,inplace=True)
        return signals
class KipnisDefensiveAdaptiveAssetAllocationStrategy(TacticalAssetAllocation):
    """
         Investment Universe: SPY, VGK, EWJ, EEM, VNQ, RWX, IEF, TLT, DBC, and GLD
         Crash Protection: IEF and Cash
         Canary: EEM and AGG
    """
    def __init__(self,
                 instruments=['SPY','VGK','EWJ','EEM','VNQ','RWX','IEF','TLT','DBC','GLD','AGG'],
                 crash_protection=['IEF'],
                 canary=['EEM', 'AGG'],
                 name='kipnis_defensive_adaptive_asset_allocation'):
        super().__init__(name=name, instruments=instruments)
        self.crash_protection = crash_protection
        self.canary = canary
        ls = self.instruments[::]
        for instrument in self.crash_protection+self.canary:
            ls.remove(instrument)
        self.risky = ls
        del ls
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        close_price = price_copy.resample('BM').last().resample('B').last().ffill().loc[:datetime.datetime.today(), :]
        momentum_score_df = momentum_score(price_copy, close_price).ffill()
        momentum_score_df = momentum_score_df.loc[price_copy.index, :]
        cov_df = cov_matrix(price_copy)
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date, series_s in momentum_score_df.iterrows():
            if is_end_business_month(date):
                universe_s = series_s[self.risky]
                top_5_positive_score_s = universe_s[universe_s>0]
                if (not top_5_positive_score_s.empty) & (top_5_positive_score_s.shape[0]>1):
                    universe_instruments_ls = top_5_positive_score_s[:min(5,top_5_positive_score_s.shape[0])].index.tolist()
                    cov_top_instruments_df = cov_df.loc[(date, universe_instruments_ls), universe_instruments_ls]
                    bnds = tuple([(0, 1) for _ in range(0, cov_top_instruments_df.shape[1])])
                    x0 = [1 / cov_top_instruments_df.shape[1] for _ in range(0, cov_top_instruments_df.shape[1])]
                    res = minimize(lambda x,cov:np.dot(x,np.dot(cov,x)),
                                   args=cov_top_instruments_df.values,
                                   constraints=({'type': 'eq', 'fun': lambda x: sum(x) - 1}),
                                   bounds=bnds,
                                   x0=x0)
                    universe_instruments_weights_s = pd.Series(index=universe_instruments_ls,data=res.x)
                    valid_canary_assets_s = series_s[self.canary].dropna()
                    if not valid_canary_assets_s.empty:
                        valid_canary_assets_s = valid_canary_assets_s[valid_canary_assets_s>0]
                        amount_of_canary_assets_positive = valid_canary_assets_s.shape[0]
                        if amount_of_canary_assets_positive == 1:
                            signals.loc[date,universe_instruments_ls] = .5*universe_instruments_weights_s
                            signals.loc[date,self.crash_protection[0]] = .5*(series_s[self.crash_protection[0]]>0)
                        elif amount_of_canary_assets_positive == 2:
                            signals.loc[date,universe_instruments_ls] = universe_instruments_weights_s
                        else:
                            signals.loc[date,self.crash_protection[0]] = 1
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class GlobalTacticalAssetAllocationStrategy(TacticalAssetAllocation):
    """
        18% SPY,13.5% EFA,4.5% EEM, 19.8% LQD, 14.4% BNDX,
        13.5% TLT, 1.8% TIP, 5% DBC, 5% GLD, 4.5% VNQ
    """
    def __init__(self,
                 instruments=['IWD','MTUM','IWM','DWAS','EFA','EEM','IEF','BWX','LQD','TLT','DBC','GLD','VNQ'],
                 name='global_tactical_asset_allocation'):
        super().__init__(name=name, instruments=instruments)
        self.holdings = pd.Series(index=self.instruments,
                                  data=[.05,.05,.05,.05,.1,.1,.05,.05,.05,.05,.1,.1,.2])
    def rule(self, price):
        price_copy = price.loc[:,self.instruments].copy()
        month = int(252/12)
        month_10_ma = price_copy.rolling(window=10*month).mean()
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date, series_s in month_10_ma.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    valid_instruments_s = series_s.dropna()
                    valid_instruments_s = price_copy.loc[date,valid_instruments_s.index.tolist()]>valid_instruments_s
                    valid_instruments_s = valid_instruments_s[valid_instruments_s==True]
                    valid_instruments_s = valid_instruments_s * self.holdings[valid_instruments_s.index]
                    signals.loc[date,valid_instruments_s.index.tolist()] = valid_instruments_s
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class QuintSwitchingFilteredStrategy(TacticalAssetAllocation):
    """
        75% IEF, 10% QQQ, 8% EEM, 4% EFA, 2% TLT, 1% SPY
    """
    def __init__(self,
                 instruments=['QQQ', 'EEM', 'EFA', 'TLT', 'SPY','IEF'],
                 risky = ['QQQ', 'EEM', 'EFA', 'TLT', 'SPY'],
                 defensive = ['IEF'],
                 name='quint_switching_filtered'):
        super().__init__(name=name, instruments=instruments)
        self.holdings = pd.Series(index=self.instruments,
                                  data=[.75,.1,.08,.04,.02,.1])
        self.risky = risky
        self.defensive = defensive
    def rule(self, price):
        price_copy = price.loc[:,self.instruments].copy()
        month = int(252/12)
        month_3_return = np.log(price_copy.div(price_copy.shift(3*month)))
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date, series_s in month_3_return.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    valid_instruments_s = series_s.dropna().copy()
                    if valid_instruments_s[valid_instruments_s>0].equals(valid_instruments_s):
                        top_3_month_return = valid_instruments_s.sort_values(ascending=False).index[0]
                        signals.loc[date,top_3_month_return] = self.holdings[top_3_month_return]
                    else:
                        signals.loc[date,self.defensive[0]] = self.holdings[self.defensive[0]]
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class CompositeDualMomentumStrategy(TacticalAssetAllocation):
    """
        25% Equities: SPY and EFA
        25% Real Estate: VNQ and REM
        25% Stress: GLD and TLT
        25% Bonds: HYG and LQD
        https://allocatesmartly.com/antonaccis-composite-dual-momentum/
    """
    def __init__(self,
                 instruments=['SPY','EFA','VNQ','REM','GLD','TLT','HYG','LQD','BIL'],
                 equities=['SPY','EFA'],
                 real_estate=['VNQ','REM'],
                 stress=['GLD','TLT'],
                 bonds=['HYG','LQD'],
                 name='composite_dual_momentum'):
        super().__init__(name=name, instruments=instruments)
        self.equities = equities
        self.real_estate = real_estate
        self.stress = stress
        self.bonds = bonds
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        month = int(252/12)
        month_12_return = np.log(price_copy.div(price_copy.shift(12 * month)))
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        price_copy.drop('BIL',axis=1,inplace=True)
        bil_s = month_12_return.pop('BIL')
        for date,series_s in month_12_return.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    if not series_s[series_s>bil_s[date]].empty:
                        top_equities = series_s[self.equities].sort_values(ascending=False).index[0]
                        top_real_estate = series_s[self.real_estate].sort_values(ascending=False).index[0]
                        top_stress = series_s[self.stress].sort_values(ascending=False).index[0]
                        top_bonds = series_s[self.bonds].sort_values(ascending=False).index[0]
                        top_per_asset_class = [top_equities,top_real_estate,top_stress,top_bonds]
                        signals.loc[date,top_per_asset_class] = .25
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class GEMDualMomentumStrategy(TacticalAssetAllocation):
    """
        45% SPY, 28% AGG, 27% EFA and BIL
    """
    def __init__(self,
                 instruments=['SPY', 'AGG', 'EFA','BIL'],
                 name='gem_dual_momentum'):
        super().__init__(name=name, instruments=instruments)
        self.holdings = pd.Series(index=self.instruments,data=[.45,.28,.27,0])

    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        month = int(252/12)
        month_12_return = np.log(price_copy.div(price_copy.shift(12 * month)))
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        price_copy.drop('BIL', axis=1, inplace=True)
        bil_s = month_12_return.pop('BIL')
        for date, series_s in month_12_return.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    if not pd.isna(bil_s[date]):
                        if bil_s[date]>series_s['SPY']:
                            signals.loc[date, 'AGG'] = self.holdings['AGG']
                        else:
                            series_s.drop('AGG', inplace=True)
                            top = series_s.sort_values(ascending=False).index[0]
                            signals.loc[date,top] = self.holdings[top]
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class DefensiveAssetAllocationStrategy(TacticalAssetAllocation):
    """
        Risky assets: SPY, IWM, QQQ, VGK, EWJ, VNQ, DBC, GLD, TLT, HYG, LQD
        Protective assets: SHY, IEF
        Canary assets: EEM, AGG
    """
    def __init__(self,
                 instruments=['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VNQ', 'DBC', 'GLD',
                              'TLT', 'LQD', 'HYG', 'SHY', 'IEF', 'EEM', 'AGG'],
                 risky=['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG'],
                 protective=['IEF', 'SHY'],
                 canary = ['EEM', 'AGG'],
                 name='defensive_asset_allocation'):
        super().__init__(name=name, instruments=instruments)
        self.risky = risky,
        self.protective = protective,
        self.canary = canary
    def rule(self, price):
        price_copy = price.loc[:,self.instruments].copy()
        close_price = price_copy.resample('BM').last().resample('B').last().ffill().loc[:datetime.datetime.today(), :]
        momentum_score_df = momentum_score(price_copy,close_price)
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date,series in momentum_score_df.iterrows():
            if is_end_business_month(date):
                if not momentum_score_df.loc[date,:].dropna().empty:
                    if not momentum_score_df.loc[date,self.canary].dropna().empty:
                        canary_asset_negative_s = momentum_score_df.loc[date,self.canary].dropna()
                        canary_asset_negative_s = canary_asset_negative_s[canary_asset_negative_s<0]
                        number_of_negative_canary = 0 if canary_asset_negative_s.empty else canary_asset_negative_s.shape[0]
                        valid_risky_instruments_s = momentum_score_df.loc[date,self.risky].dropna()
                        valid_protective_instruments_s = momentum_score_df.loc[date, self.protective].dropna()
                        top_risky_instruments_ls = \
        valid_risky_instruments_s.sort_values(ascending=False).index[:min(6,valid_risky_instruments_s.shape[0])].tolist()
                        signals.loc[date,top_risky_instruments_ls] = 1
                        top_protective_instrument = \
        valid_protective_instruments_s.sort_values(ascending=False).index[0]
                        signals.loc[date,top_protective_instrument] = 1
                        signals.loc[date,top_risky_instruments_ls] = (1-number_of_negative_canary/2)/valid_risky_instruments_s.shape[0]
                        signals.loc[date,top_protective_instrument] = number_of_negative_canary*signals.loc[date,top_protective_instrument]
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class ProtectiveAssetAllocationStrategy(TacticalAssetAllocation):
    """
        51% IEF, 6% IWM, 6% QQQ, 5% VNQ, 5% SPY, 5% VGK, 5% EEM, 4% EWJ, 3% DBC,\
        3% TLT, 3% GLD, 2% HYG and 2% LQD
    """
    def __init__(self,
                 instruments=['IWM','QQQ','VNQ','SPY','VGK','EEM','EWJ','DBC','TLT','GLD','HYG','LQD','IEF'],
                 risky=['IWM','QQQ','VNQ','SPY','VGK','EEM','EWJ','DBC','TLT','GLD','HYG','LQD'],
                 safe=['IEF'],
                 name='protective_asset_allocation'):
        super().__init__(name=name, instruments=instruments)
        self.risky = risky
        self.safe = safe
        self.holdings = pd.Series(index=self.safe+self.risky,
                                  data=[.51,.06,.06,.05,.05,.05,.05,.04,.03,.03,.03,.02,.02])
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        momentum_score_sma = price_copy.div(price_copy.rolling(window=252).mean())-1
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        for date,series_s in momentum_score_sma.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    number_of_funds_with_MOM_positive_s = series_s.copy()
                    number_of_funds_with_MOM_positive_s.dropna(inplace=True)
                    number_of_funds_with_MOM_positive = \
                        number_of_funds_with_MOM_positive_s[number_of_funds_with_MOM_positive_s>0].shape[0]
                    number_of_funds_with_MOM_positive = min(number_of_funds_with_MOM_positive,12)
                    signals.loc[date,self.safe[0]] = \
(number_of_funds_with_MOM_positive <= 6)*1+(number_of_funds_with_MOM_positive >= 7)*(12-number_of_funds_with_MOM_positive)/6
                    top_risky_instruments_s = \
        series_s[self.risky].sort_values(ascending=False)[:min(6,series_s.shape[0])].copy()
                    top_risky_instruments_s.dropna(inplace=True)
                    signals.loc[date, top_risky_instruments_s.index.tolist()] = \
(number_of_funds_with_MOM_positive >= 7)*(number_of_funds_with_MOM_positive/min(6,top_risky_instruments_s.shape[0]))
        signals = signals*self.holdings
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class TrendIsYourFriendStrategy(TacticalAssetAllocation):
    """
        20% Equities, 26% Bonds, and 54% Cash, Commodities and Real Estate
    """
    def __init__(self,
                 instruments=['SPY','BIL','VNQ','DBC'],
                 equities=['SPY'],
                 bonds=['BIL'],
                 CCRE=['VNQ','DBC'],
                 name='trend_is_our_friend'):
        super().__init__(name=name, instruments=instruments)
        self.equities = equities
        self.bonds = bonds
        self.CCRE = CCRE
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        month = int(252/12)
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        month_10_ma_df = price_copy.rolling(window=10*month).mean()
        returns_df = np.log(price_copy.div(price_copy.shift()))
        vol_df = vol_estimate(returns_df)
        for date,series_s in month_10_ma_df.iterrows():
            if is_end_business_month(date):
                sigma_t = vol_df.loc[date,:]
                sigma_t.dropna(inplace=True)
                series_s.dropna(inplace=True)
                if not (sigma_t.empty | series_s.empty):
                    k_t = 1/sigma_t
                    w_it = k_t*sigma_t
                    valid_instruments_s = price_copy.loc[date,series_s.index.tolist()]
                    valid_instruments_s = valid_instruments_s[valid_instruments_s>series_s]
                    if not valid_instruments_s.empty:
                        signals.loc[date,valid_instruments_s.index.tolist()] = w_it
        signals.replace(np.nan,0,inplace=True)
        return signals.values
class GeneralizedProtectiveMomentumStrategy(TacticalAssetAllocation):
    """
        Risk:  SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, DBC, GLD, HYG, and LQD
        Safety: BIL, IEF
    """
    def __init__(self,
                 instruments=['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'HYG', 'LQD','BIL','IEF'],
                 risky=['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'HYG', 'LQD'],
                 safe=['BIL','IEF'],
                 name='generalised_protective_momentum'):
        super().__init__(name=name, instruments=instruments)
        self.risky = risky
        self.safe = safe
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        average_returns_df = ri(price_copy)
        corr_df = ci(price_copy)
        for date,series_s in average_returns_df.iterrows():
            if is_end_business_month(date):
                ci_s = corr_df.loc[(date,'equally_weighted'),:].drop('equally_weighted')
                if not(series_s.dropna().empty|ci_s.dropna().empty):
                    ri_times_one_minus_ci = series_s * (1 - ci_s)
                    top_safe_assets_s = ri_times_one_minus_ci[self.safe].dropna()
                    if not top_safe_assets_s.empty:
                        top_safe_assets = top_safe_assets_s.sort_values(ascending=False).index[0]
                        valid_risky_instruments_s = ri_times_one_minus_ci.loc[self.risky].dropna().copy()
                        number_of_risky_assets_positive = valid_risky_instruments_s[valid_risky_instruments_s>0].shape[0]
                        top_risky_assets_s = valid_risky_instruments_s.sort_values(ascending=False)
                        top_risky_assets_s = top_risky_assets_s[:min(3,valid_risky_instruments_s.shape[0])]
                        signals.loc[date,top_risky_assets_s.index.tolist()] = \
                            (1/top_risky_assets_s.shape[0])*(number_of_risky_assets_positive/6)
                        signals.loc[date,top_safe_assets] = (12-number_of_risky_assets_positive)/6
                        signals.loc[date,top_safe_assets] = \
(number_of_risky_assets_positive>6)*signals.loc[date,top_safe_assets]+(number_of_risky_assets_positive<=6)*1
                        signals.loc[date, top_risky_assets_s.index.tolist()] = \
                            (number_of_risky_assets_positive > 6) * signals.loc[date, top_risky_assets_s.index.tolist()]
        signals.replace(np.nan,0,inplace=True)
        return signals.values

class AdaptiveAssetAllocationStrategy(TacticalAssetAllocation):
    """
        IEF,SPY,VNQ,RWX,DBC,EEM,VGK,TLT,GLD,EWJ
    """
    def __init__(self,
                 instruments=['IEF','SPY','VNQ','RWX','DBC','EEM','VGK','TLT','GLD','EWJ'],
                 name='adaptive_asset_allocation'):
        super().__init__(name=name, instruments=instruments)
    def rule(self, price):
        price_copy = price.loc[:, self.instruments].copy()
        signals = pd.DataFrame(index=price_copy.index, columns=price_copy.columns, data=np.nan)
        month = int(252/12)
        returns_6month_df = np.log(price_copy.div(price_copy.shift(6*month)))
        vol_df = np.log(price_copy.div(price_copy.shift(1 * month))).expanding().std()
        cov_df = returns_6month_df.expanding().corr()
        for date,series_s in returns_6month_df.iterrows():
            if is_end_business_month(date):
                if not series_s.dropna().empty:
                    valid_instruments_s = series_s.dropna()
                    vol_valid_instruments_s = vol_df.loc[date,:].dropna()[valid_instruments_s.index]
                    top_instruments_ls = \
                        valid_instruments_s.sort_values(ascending=False).index[:min(5,valid_instruments_s.shape[0])]
                    cov_top_instruments_df = \
                        cov_df.loc[(date,top_instruments_ls),top_instruments_ls]
                    # Change vol in the covariance matrix with the
                    np.fill_diagonal(cov_top_instruments_df.values,vol_df.loc[date,:].dropna().values)
                    cov_top_instruments_df = cov_top_instruments_df**2
                    bnds = tuple([(0,1) for _ in range(0,cov_top_instruments_df.shape[1])])
                    x0 = [1/cov_top_instruments_df.shape[1] for _ in range(0,cov_top_instruments_df.shape[1])]
                    res = minimize(lambda x,cov: np.dot(x,np.dot(cov,x)),args=(cov_top_instruments_df.values),
                                   x0=x0,
                                   constraints=({'type':'eq','fun':lambda x:sum(x)-1}),
                                   bounds=bnds)
                    signals.loc[date,cov_top_instruments_df.columns.tolist()] = res.x
        signals.replace(np.nan,0,inplace=True)
        return signals.values


if __name__ == '__main__':
    pass
    # data_obj = Data()
    # query = data_obj.write_query_price()
    # df = data_obj.query(query,set_index=True)
    # strategy_obj = KipnisDefensiveAdaptiveAssetAllocation()
    # signals = strategy_obj.rule(df)
    # entries = signals > 0
    # size = signals
    # pf = vbt.Portfolio.from_signals(close=df.loc[:,strategy_obj.instruments],
    #                                 entries=entries,
    #                                 size=size,
    #                                 size_type='Percent',
    #                                 group_by=True,
    #                                 init_cash=10000,
    #                                 cash_sharing=True,
    #                                 freq='d')
    # pdb.set_trace()
