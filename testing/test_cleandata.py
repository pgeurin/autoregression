import unittest
import pandas as pd
from autoregression.cleandata import rename_columns

class TestClean(unittest.TestCase):
    def test_rename_columns(self):
        df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['First', 'Second_and THIRD', '__Fourth', 'FIF.TH aND sixth'])
        df_snake = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['first', 'second_and_third', '_fourth', 'fif_th_and_sixth'])
        print(df)
        print(df_snake)
        df_rename = rename_columns(df)
        print(df_rename)
        self.assertTrue(all(x == y
                            for x, y in zip(df_snake.columns, df_rename.columns)))
        # self.assertEqual(all(x == y
        #                      for x, y in zip(df_snake.values, df_rename.values)))
        # self.assertTrue(all(x == y
        #                     for x, y in zip(df_snake.index, df_rename.index)))
        # self.assertTrue(df_snake.equals(df_rename))

if __name__ == "__main__":
    unittest.main()
