import plotly.graph_objects as go
import plotly.express as px
from src.tradingDashboard.data import Data
from dash import Dash, Output, Input, dcc, html
from plotly.subplots import make_subplots
from src.tradingDashboard.Strategies import GoldenButterflyStrategy
from src.tradingDashboard.Strategies import RobArmottStrategy
from src.tradingDashboard.Strategies import GlobalAssetAllocationStrategy
from src.tradingDashboard.Strategies import PermanentStrategy
from src.tradingDashboard.Strategies import DesertStrategy
from src.tradingDashboard.Strategies import LarryStrategy
from src.tradingDashboard.Strategies import BigRocksStrategy
from src.tradingDashboard.Strategies import SandwichStrategy
from src.tradingDashboard.Strategies import BalancedTaxAwareStrategy
from src.tradingDashboard.Strategies import BalancedStrategy
from src.tradingDashboard.Strategies import IncomeGrowthStrategy
from src.tradingDashboard.Strategies import IncomeGrowthTaxStrategy
from src.tradingDashboard.Strategies import ConservativeIncomeStrategy
from src.tradingDashboard.Strategies import ConservativeIncomeTaxStrategy
from src.tradingDashboard.Strategies import AllWeatherStrategy
from src.tradingDashboard.Strategies import US6040Strategy
from src.tradingDashboard.Strategies import IvyStrategy
from src.tradingDashboard.Strategies import RobustAssetAllocationBalancedStrategy
from src.tradingDashboard.Strategies import DiversifiedGEMDualMomentumStrategy
from src.tradingDashboard.Strategies import VigilantAssetAllocationG12Strategy
from src.tradingDashboard.Strategies import VigilantAssetAllocationG4Strategy
from src.tradingDashboard.Strategies import KipnisDefensiveAdaptiveAssetAllocationStrategy
from src.tradingDashboard.Strategies import GlobalTacticalAssetAllocationStrategy
from src.tradingDashboard.Strategies import QuintSwitchingFilteredStrategy
from src.tradingDashboard.Strategies import CompositeDualMomentumStrategy
from src.tradingDashboard.Strategies import GEMDualMomentumStrategy
from src.tradingDashboard.Strategies import DefensiveAssetAllocationStrategy
from src.tradingDashboard.Strategies import ProtectiveAssetAllocationStrategy
from src.tradingDashboard.Strategies import TrendIsYourFriendStrategy
from src.tradingDashboard.Strategies import GeneralizedProtectiveMomentumStrategy
from src.tradingDashboard.Strategies import AdaptiveAssetAllocationStrategy
import vectorbt as vbt
import os
import pickle
import pandas as pd
from pathlib import Path
import asyncio
import concurrent.futures
import time
import sys,inspect
import pdb

if __name__ == '__main__':
    strategies_ls = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    strategies_ls = [(pair[0], pair[1]) for pair in strategies_ls if ('Strategy' in pair[0])]
    for pair in strategies_ls:
        for allocation in ['buy_and_hold', 'tactical_asset_allocation']:
            allocation_ls = list()
            strategy = pair[1]()
            if strategy.allocation == allocation: allocation_ls.append(strategy)
    pdb.set_trace()