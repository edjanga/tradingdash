import pandas as pd
from Config import TiingoConfig
import concurrent.futures
import requests
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from Logger import Logs
import pdb
import sqlite3 as sql
import numpy as np
import os


class Data:


    conn_obj = sql.connect(os.path.abspath('/Users/emmanueldjanga/wifeyAlpha/DataStore/etfs.db'),\
                           check_same_thread=False)
    log_obj = Logs()
    tiingo_config_obj = TiingoConfig(os.path.abspath('/Users/emmanueldjanga/wifeyAlpha/Config/config_platform.json'))
    universe_ls = ['SHY','TLT','VTI','IWN','GLD','BNDX','LQD','VEU','VNQ','SPY','TIP','DBC',\
        'EFA','EEM','BIL','IEF','DLS','AGG','IWD','IWM','EFV','SCZ','HYG','JPST']

    def __init__(self,startDate=datetime.now()):
        self.startDate = startDate - relativedelta(years=25)
        self.startDate = self.startDate.strftime(format='%Y-%m-%d')
        self.endDate = datetime.today().strftime(format='%Y-%m-%d')

    def simulation(self,columns_ls=universe_ls,freq='M'):
        index_ls = pd.date_range(start=self.startDate, end=self.endDate)
        simulation_df = pd.DataFrame(columns=columns_ls,\
                                     index=index_ls,\
                                     data=np.random.normal(loc=100,size=(len(index_ls),len(columns_ls))))
        simulation_df = simulation_df.resample(freq).agg('last')
        return simulation_df

    def wait_between_query(func):
        def wrap(self):
            """
                Decorator aiming to avoid being maxed out between queries
            """
            try:
                func(self)
            except requests.ConnectionError:
                msg = f'[FETCHING LIMIT]: {func.__name__} has reached its capacity. Waiting for 1 sec.'
                self.logger_obj.log_msg(msg)
                time.sleep(1)
                func(self)
        return wrap

    @wait_between_query
    def insert_historical_data(self,freq='monthly'):
        universe_ls = Data.universe_ls
        historical_data_ls = []

        def query(sym,api,freq):
            """
            :param sym: Ticker to be queried
            :param api: Tiingo API
            :return: sym
            """
            msg = f"[FETCHING]: {sym} price data started @ {Data.log_obj.now_date()}.\n"
            Data.log_obj.log_msg(msg)
            startDate = self.startDate
            endDate = self.endDate
            #'https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=2012-1-1&endDate=2016-1-1 '
            query = f'daily/{sym}/prices?startDate={startDate}&token={api}&endDate={endDate}&resampleFreq={freq}'
            url = ''.join((self.tiingo_config_obj.endpoint, query))
            r = requests.get(url, headers=self.tiingo_config_obj.headers)
            if r.status_code == 200:
                try:
                    data_dd = r.json()[0]
                    data_dd['ticker'] = sym
                    historical_data_ls.append(data_dd)
                except IndexError:
                    msg = f'[FETCHING]: ERROR in retrieving {sym} @ {Data.log_obj.now_date()}.\n'
                    Data.log_obj.log_msg(msg)
                return sym
        # Multithreading to speed up process
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = []
            for sym in universe_ls:
                results.append(executor.submit(query,sym,Data.tiingo_config_obj.api,'daily'))
            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[FETCHING]: {f.result().upper()} price data has been retrieved @ {Data.log_obj.now_date()}.\n'
                    Data.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        historical_data_df = pd.DataFrame(historical_data_ls)
        historical_data_df =pd.pivot(historical_data_df[['adjClose', 'ticker', 'date']],\
            index='date', columns='ticker',values='adjClose').sort_index()
        historical_data_df.columns.name = None
        historical_data_df.index = [date.replace('T',' ').replace('Z','') for date in historical_data_df.index]
        historical_data_df.index = pd.to_datetime(historical_data_df.index)
        pdb.set_trace()
        historical_data_df.to_sql(con=Data.conn_obj,name='price',if_exists='replace')
        msg = f'[INSERTION]: ETF prices data ahas been inserted into the database @ {Data.log_obj.now_date()}.\n'
        self.logger_obj.log_msg(msg)
        return historical_data_df

    def write_query_strategies(self,allocation="buy_and_hold"):
        query = f'PRAGMA table_info(\"{allocation}\");'
        return query

    def write_query_symbol(self,symbol):
        if isinstance(symbol,str):
            symbol = [symbol]
        symbol = [f'\"{sym}\"' for sym in symbol]
        symbol = ','.join(symbol)
        query = f'SELECT {symbol} FROM etfs;'
        return query

    def write_query_allocation(self):
        return 'SELECT name FROM \"sqlite_master\" WHERE type = \"table\" AND name NOT LIKE \"sqlite_%\" AND name NOT LIKE \"%_returns\" AND name NOT LIKE \"%_performance\";'

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
        df = pd.read_sql(sql=query,con=Data.conn_obj)
        if melt:
            df = pd.melt(df,id_vars='index',var_name='strategy',value_name='equity_curve')
        if set_index:
            df = df.set_index('index')
        return df

    def close(self):
        Data.conn_obj.close()


if __name__ == '__main__':
    data_obj = Data()
    #query = data_obj.write_query_equity_curves()
    query = data_obj.write_query_returns()
    pdb.set_trace()
    df = data_obj.query(query=query)
    #df = data_obj.simulation()
    #allocation_query = data_obj.write_query_allocation()
    #allocation_ls = data_obj.query(query=allocation_query).name.tolist()
    data_obj.close()

    #data_obj.insert_historical_data()
    #df = data_obj.simulation()
    #pdb.set_trace()