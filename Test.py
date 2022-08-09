from unittest import TestCase
from DataStore import Data
import pdb

class TestData(TestCase):

    data_obj = Data()

    def test_write_query_equity_curves_buy_and_hold(self,allocation = 'buy_and_hold'):
        query_target = f'SELECT * FROM \"{allocation}\";'
        self.assertEqual(TestData.data_obj.write_query_equity_curves(allocation=allocation),query_target)

    def test_write_query_equity_curves_tactical_allocation(self,allocation='tactical_allocation'):
        query_target = f'SELECT * FROM \"{allocation}\";'
        self.assertEqual(TestData.data_obj.write_query_equity_curves(allocation=allocation),query_target)


if __name__ == '__main__':
    test_data_obj = TestData()
    test_data_obj.test_write_query_equity_curves_buy_and_hold(allocation='buy_and_hold')
    test_data_obj.test_write_query_equity_curves_buy_and_hold(allocation='tactical_allocation')

