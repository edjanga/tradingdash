from unittest import TestCase
from DataStore import Data
import pdb

class TestData(TestCase):

    data_obj = Data()

    def test_write_query_equity_curves_buy_and_hold(self,allocation='buy_and_hold'):
        query_target = f'SELECT * FROM \"{allocation}\";'
        self.assertEqual(TestData.data_obj.write_query_equity_curves(allocation=allocation),query_target)

    def test_write_query_equity_curves_tactical_allocation(self,allocation='tactical_allocation'):
        query_target = f'SELECT * FROM \"{allocation}\";'
        self.assertEqual(TestData.data_obj.write_query_equity_curves(allocation=allocation),query_target)

    def test_write_query_returns(self,allocation='buy_and_hold'):
        query_target = f'SELECT * FROM \"{allocation}_returns\";'
        self.assertEqual(TestData.data_obj.write_query_returns(allocation=allocation), query_target)

    def test_write_query_performance(self,allocation='buy_and_hold'):
        query_target = f'SELECT * FROM \"{allocation}_performance\";'
        self.assertEqual(TestData.data_obj.write_query_performance(allocation=allocation), query_target)

    def test_write_query_symbol_single(self,symbol='BIL'):
        query_target = f'SELECT \"{symbol}\",\"index\" FROM price;'
        self.assertEqual(TestData.data_obj.write_query_symbol(symbol=symbol),query_target)

    def test_write_query_symbol_multiple(self,symbol=['BIL','JPST','VNQ']):
        symbol_str = ','.join([f'\"{sym}\"' for sym in symbol])
        query_target = f'SELECT {symbol_str},\"index\" FROM price;'
        self.assertEqual(TestData.data_obj.write_query_symbol(symbol=symbol),query_target)

    def test_write_query_strategies(self,allocation='buy_and_hold'):
        query_target = f'PRAGMA table_info(\"{allocation}\");'
        self.assertEqual(TestData.data_obj.write_query_strategies(),query_target)


if __name__ == '__main__':
    test_data_obj = TestData()
    test_data_obj.test_write_query_equity_curves_buy_and_hold(allocation='buy_and_hold')
    test_data_obj.test_write_query_equity_curves_buy_and_hold(allocation='tactical_allocation')
    test_data_obj.test_write_query_returns(allocation='buy_and_hold')
    test_data_obj.test_write_query_returns(allocation='tactical_allocation')
    test_data_obj.test_write_query_performance(allocation='buy_and_hold')
    test_data_obj.test_write_query_symbol_single()
    test_data_obj.test_write_query_symbol_multiple()
    test_data_obj.test_write_query_strategies(allocation='buy_and_hold')

