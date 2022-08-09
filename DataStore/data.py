import pandas as pd
from Config import TiingoConfig
import concurrent.futures
import requests
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from Logger import Logs



class Data:

    log_obj = Logs()
    tiingo_config_obj = TiingoConfig('../Config/config_platform.json')
    universe_ls = ['SHY','TLT','VTI','IWN','GLD','BNDX','LQD','VEU','VNQ','SPY','TIP','DBC',\
        'EFA','EEM','BIL','IEF','DLS','AGG','IWD','IWM','EFV','SCZ','HYG']

    def __init__(self,startDate=datetime.today()):
        self.startDate = startDate - relativedelta(years=25)

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
    #@insert_equities_historical_data_condition
    def insert_historical_data(self):
        universe_ls = Data.universe_ls
        historical_data_ls = []

        def query(sym, api):
            """
            :param sym: Ticker to be queried
            :param api: Tiingo API
            :return: sym
            """
            msg = f"[FETCHING]: {sym} price data started @ {Data.log_obj.now_date()}.\n"
            Data.log_obj.log_msg(msg)
            startDate = self.startDate.split(' ')[0]
            query = f'daily/{sym}/prices?startDate={startDate}&token={api}'
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
                results.append(executor.submit(query,sym,Data.tiingo_config_obj.api))

            for f in concurrent.futures.as_completed(results):
                try:
                    msg = f'[FETCHING]: {f.result().upper()} price data has been retrieved @ {Data.log_obj.now_date()}.\n'
                    Data.log_obj.log_msg(msg)
                except AttributeError:
                    continue
        historical_data_df = pd.DataFrame(historical_data_ls)
