import pandas as pd


def returns(df):
    return df.pct_change()

def equity_curve(returns_df,weights_df):
    weights_df = weights_df.shift()
    equity_curve_df = pd.DataFrame(weights_df.mmult(returns_df).sum(axis=1))
    return equity_curve_df

class BuyAndHold:

    @staticmethod
    def golden_butterfly(df):
        """
            40% fixed income (20% SHY, 20% TLT)
            40% equities (20% VTI, 20% IWN)
            20% gold (GLD)
        """
        universe_df = df.filer(regex=r'(SHY|TLT|VTI|IWN|GLD)')
        returns_df = returns(df)
        weights_ls = [.2] * returns.shape[1]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df,weights_df)
        return equity_curve_df

    @staticmethod
    def rob_armott(df):
        """
            20% BNDX,20% LQD,10% VEU,10% VNQ,
            10% SPY,10% TLT,10% TIP,10% DBC
        """
        universe_df = df.filer(regex=r'(BNDX|LQD|VEU|VNQ|SPY|TLT|TIP|DBC)')
        returns_df = returns(universe_df)
        weights_ls = [.2,.2,.1,.1,.1,.1,.1,.1]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def global_asset_allocation(df):
        """
            18% SPY,13.5% EFA,4.5% EEM, 19.8% LQD, 14.4% BNDX,
            13.5% TLT, 1.8% TIP, 5% DBC, 5% GLD, 4.5% VNQ
        """
        universe_df = df.filer(regex=r'(SPY|EFA|EEM|LQD|BNDX|TLT|TIP|DBC|GLD|VNQ)')
        returns_df = returns(universe_df)
        weights_ls = [.18,.135,.045,.198,.144,.135,.018,.05,.045]
        assert sum(weights_ls)==1
        weights_df = pd.DataFrame(index=returns_df.index,columns=universe_df.columns,data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def permanent(df):
        """
            25% BIL,25% GLD,25% TLT and 25% SPY
        """
        universe_df = df.filer(regex=r'(BIL|GLD|TLT|SPY)')
        returns_df = returns(universe_df)
        weights_ls = [.25] * returns_df.shap[1]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def desert(df):
        """
            60% IEF, 30% VTI, 10% GLD
        """
        universe_df = df.filer(regex=r'(IEF|VTI|GLD)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.3,.1]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def larry(df):
        """
            30% in equities  (15% IWN, 7.5% IWN, 7.5% EEM)
            70% in bonds (IEF)
        """
        universe_df = df.filer(regex=r'(IWN|IWN|EEM|LQD|IEF)')
        returns_df = returns(universe_df)
        weights_ls = [.15,.075,.075,.7]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df

    @staticmethod
    def big_rocks(df):
        """
            60% AGG, 6% SPY, 6% IWD, 6% IWM, 6% IWN, 4% EFV, 4% VNQ, 2% EFA, 2% SCZ, 2% DLS, 2% EEM
        """
        universe_df = df.filer(regex=r'(AGG|SPY|IWD|IWM|IWN|EFV|VNQ|EFA|SCZ|DLS|EEM)')
        returns_df = returns(universe_df)
        weights_ls = [.6,.06,.06,.06,.06,.04,.04,.02,.02,.02,.02]
        assert sum(weights_ls) == 1
        weights_df = pd.DataFrame(index=returns_df.index, columns=universe_df.columns, data=weights_ls)
        equity_curve_df = equity_curve(returns_df, weights_df)
        return equity_curve_df



class TacticalAllocation:

    pass
