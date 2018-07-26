import unittest
import pandas as pd
import numpy as np
from autoregression.cleandata import rename_columns

class TestClean(unittest.TestCase):
    def test_rename_columns(self):
        df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['First', 'Second_and THIRD', '__Fourth', 'FIF.TH aND sixth'])
        df_snake = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['first', 'second_and_third', '_fourth', 'fif_th_and_sixth'])
        df_rename = rename_columns(df)
        self.assertTrue(all(x == y
                            for x, y in zip(df_snake.columns, df_rename.columns)))
        # NOTE: Indexes aren't the same bc one is indexed strings '0', '1' and the other is ints 0, 1
        # print(df)
        # print(df_snake)
        # print(df_rename)
        # print(df_snake.index)
        # print(df_rename.index)
        # print(list(df_snake.index))
        # print(list(df_rename.index))
        #
        # self.assertTrue(np.testing.assert_array_equal(df_snake.values, df_rename.values))
        self.assertTrue((df_snake.values == df_rename.values).all())
        # self.assertTrue(list(df_snake.index) == list(df_rename.index))
        # self.assertTrue(df_snake.equals(df_rename))

if __name__ == "__main__":
    unittest.main()
