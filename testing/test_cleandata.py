import unittest
import pandas as pd
import numpy as np
from autoregression.cleandata import rename_columns, add_feature_continuous_condition
from nose.tools import assert_dict_equal

class TestClean(unittest.TestCase):
    def test_rename_columns(self):
        df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['First', 'Second_and THIRD', '__Fourth', 'FIF.TH aND sixth'])
        df_snake = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=['first', 'second_and_third', '_fourth', 'fif_th_and_sixth'])
        df_rename = rename_columns(df)
        self.assertTrue(all(x == y
                            for x, y in zip(df_snake.columns, df_rename.columns)))
        assert_dict_equal(df_snake.to_dict(), df_rename.to_dict())
        # self.assertTrue((df_snake.values == df_rename.values).all())
        # NOTE: Indexes aren't the same bc one is indexed strings '0', '1' and the other is ints 0, 1
        # print(df)
        # print(df_snake)
        # print(df_rename)
        # print(df_snake.index)
        # print(df_rename.index)
        # print(list(df_snake.index))
        # print(list(df_rename.index))
        # self.assertTrue(np.testing.assert_array_equal(df_snake.values, df_rename.values))
        # self.assertTrue(list(df_snake.index) == list(df_rename.index))
        # self.assertTrue(df_snake.equals(df_rename))


    # def test_add_feature_continuous_condition(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=1)
    #     df_X_exp = pd.DataFrame([[1, 2, True], [1, 2, True]], columns=['first', 'second', '==_first'])
    #     self.assertTrue(df_X_aft.to_dict(), df_X_exp.to_dict())
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=1)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=1)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='<', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='>', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=2)
    #
    #     df_X = pd.DataFrame([[1, 2, 3, 4], [np.NaN, 2, 3, 4]], columns=['first', 'second_and_third', 'fourth', 'fifth_and_sixth'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=1)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=1)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='<', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='>', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=2)
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=2)


if __name__ == "__main__":
    unittest.main()
