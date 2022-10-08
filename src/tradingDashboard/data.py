import pandas as pd
import concurrent.futures
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.tradingDashboard.logger import Logs
import sqlite3 as sql
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv,find_dotenv
import pdb
import vectorbt as vbt
from scipy.stats import skew,kurtosis

load_dotenv(find_dotenv(Path('../.env')))

class Data:

    log_obj = Logs()

    universe_ls = ['SHY','TLT','VTI','IWN','GLD','BNDX','LQD','VEU','VNQ','SPY','TIP','DBC',\
    'EFA','EEM','BIL','IEF','DLS','AGG','IWD','IWM','EFV','SCZ','HYG','NEAR','QQQ','MTUM','IWB','IEFA',\
    'DWAS','BWX','VGK','EWJ','REM','RWX','UVIX','SVIX']

    def __init__(self,startDate=datetime.now()):
        self.startDate = startDate - relativedelta(years=25)
        self.startDate = self.startDate.strftime(format='%Y-%m-%d')
        self.endDate = datetime.today().strftime(format='%Y-%m-%d')
        try:
            if 'etfs.db' not in os.listdir():
                raise sql.OperationalError
            else:
                self.conn_obj = sql.connect(database=Path(f'{os.curdir}/etfs.db'),check_same_thread=False)
        except sql.OperationalError:
            self.conn_obj = sql.connect(database=Path('./src/tradingDashboard/etfs.db'), check_same_thread=False)


    def simulation(self,columns_ls=universe_ls,freq='M'):
        """
            Dummy method used to generate a dataframe with random numbers normally distributed (proof of concept).
        """
        index_ls = pd.date_range(start=self.startDate, end=self.endDate)
        simulation_df = pd.DataFrame(columns=columns_ls,\
                                     index=index_ls,\
                                     data=np.random.normal(loc=100,size=(len(index_ls),len(columns_ls))))
        simulation_df = simulation_df.resample(freq).agg('last')
        return simulation_df

    def insert_historical_data(self,freq='daily'):
        universe_ls = Data.universe_ls
        historical_data_ls = []

        def query(sym,freq):
            """
            :param sym: Ticker to be queried
            :param api: Tiingo API
            :return: sym
            """
            msg = f"[FETCHING]: {sym} price data started @ {Data.log_obj.now_date()}.\n"
            Data.log_obj.log_msg(msg)
            startDate = self.startDate
            endDate = self.endDate
            query = f'daily/{sym.lower()}/prices?startDate={startDate}&token={os.environ.get("TIINGO_API")}&endDate={endDate}&resampleFreq={freq}'
            url = ''.join((os.environ.get('TIINGO_ENDPOINT'), query))
            r = requests.get(url, headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                try:
                    data_dd = r.json()
                    sym_df = pd.DataFrame(data_dd)
                    price_s = sym_df.adjClose
                    price_s.name = sym
                    price_s.index = sym_df.date
                    historical_data_ls.append(price_s)
                except IndexError:
                    msg = f'[FETCHING]: ERROR in retrieving {sym} @ {Data.log_obj.now_date()}.\n'
                    Data.log_obj.log_msg(msg)
                return sym
        # Multithreading to speed up process
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for sym in universe_ls:
                results.append(executor.submit(query, sym, freq))
            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[FETCHING]: {f.result().upper()} price data has been retrieved @ {Data.log_obj.now_date()}.\n'
                    Data.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        historical_data_df = pd.DataFrame(historical_data_ls).transpose().sort_index().dropna(how='all', axis=0)
        historical_data_df.index = [date.replace('T',' ').replace('Z','') for date in historical_data_df.index]
        historical_data_df.index = pd.to_datetime(historical_data_df.index)
        historical_data_df.to_sql(con=self.conn_obj,name='price',if_exists='replace')
        msg = f'[INSERTION]: ETF prices data ahas been inserted into the database @ {Data.log_obj.now_date()}.\n'
        Data.log_obj.log_msg(msg)

    def write_query_price(self):
        query = 'SELECT * FROM price;'
        return query

    def write_query_strategies(self,allocation="buy_and_hold"):
        query = f'PRAGMA table_info(\"{allocation}\");'
        return query

    def write_query_symbol(self,symbol):
        if isinstance(symbol,str):
            symbol = [symbol]
        symbol = symbol + ['index']
        symbol = [f'\"{sym}\"' for sym in symbol]
        symbol = ','.join(symbol)
        query = f'SELECT {symbol} FROM price;'
        return query

    def write_query_allocation(self):
        return 'SELECT name FROM \"sqlite_master\" WHERE type = \"table\" AND name NOT LIKE \"sqlite_%\" AND name NOT LIKE \"%_returns\" AND name NOT LIKE \"%_performance\" AND name NOT LIKE \"%price\";'

    def write_query_equity_curves(self,allocation='buy_and_hold'):
        query = f'SELECT * FROM \"{allocation}\";'
        return query

    def write_query_returns(self,allocation='buy_and_hold'):
        table = '_'.join((allocation,'returns'))
        query = f'SELECT * FROM \"{table}\";'
        return query

    def write_query_performance(self,allocation='buy_and_hold'):
        table = '_'.join((allocation,'performance'))
        query = f'SELECT * FROM \"{table}\";'
        return query

    def query(self,query,melt=False,set_index=False):
        df = pd.read_sql(sql=query,con=self.conn_obj)
        if melt:
            df = pd.melt(df,id_vars='index',var_name='strategy',value_name='equity_curve')
        if set_index:
            df = df.set_index('index')
            df.index = pd.to_datetime(df.index)
        return df

    def close(self):
        self.conn_obj.close()


if __name__ == '__main__':
    data_obj = Data()
    #query = "SELECT buy_and_hold_returns.\"index\", buy_and_hold_returns.average, tactical_allocation_returns.average FROM buy_and_hold_returns INNER JOIN tactical_allocation_returns ON buy_and_hold_returns.\"index\" = tactical_allocation_returns.\"index\";"
    ##df = data_obj.query(query=query,set_index=True)
    #df.columns = ['buy_and_hold','tactical_asset_allocation']
    #df['cross_asset_allocation'] = df.mean(axis=1)
    #df = pd.DataFrame(df.cross_asset_allocation)
    #df.to_sql(name='cross_asset_allocation_returns',con=data_obj.conn_obj,if_exists='replace')
    perf_dd = dict()
    query = data_obj.write_query_equity_curves(allocation='cross_asset_allocation')
    df = data_obj.query(query,set_index=True)
    ret = df.cross_asset_allocation[-1] / df.cross_asset_allocation[0]
    ret = 100*(ret**(1/25)-1)
    perf_dd['Annualized Return [%]'] = [ret]
    query = data_obj.write_query_returns(allocation='cross_asset_allocation')
    df = data_obj.query(query, set_index=True)
    perf_dd['Annualized Volatility [%]'] = [df.cross_asset_allocation.std()*252**(1/2)]
    perf_dd['Max Drawdown [%]'] = [vbt.returns.nb.max_drawdown_1d_nb(df.cross_asset_allocation.values)*100]
    perf_dd['Sharpe Ratio'] = [vbt.returns.nb.sharpe_ratio_1d_nb(df.cross_asset_allocation.values,ann_factor=252**(.5))]
    perf_dd['Skew'] = [skew(df.cross_asset_allocation.values)]
    perf_dd['Kurtosis'] = [kurtosis(df.cross_asset_allocation.values)]
    perf_dd['CVar'] = \
        [df.cross_asset_allocation[df.cross_asset_allocation<=df.cross_asset_allocation.quantile(q=.05)].mean()]
    perf_df = pd.DataFrame(perf_dd,index=['cross_asset_allocation'])
    perf_df.to_sql(con=data_obj.conn_obj,name='cross_asset_allocation_performance',if_exists='replace')
    pdb.set_trace()
    #data_obj.insert_historical_data()
    #data_obj.close()