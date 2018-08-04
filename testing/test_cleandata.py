import unittest
import pandas as pd
import numpy as np
from autoregression.cleandata import rename_columns, add_feature_continuous_condition, clean_df_X, clean_df
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


    # def test_add_feature_continuous_condition_equal_1(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=1)
    #     df_X_exp = pd.DataFrame([[1, 2, True], [1, 2, True], [0, 3, False]], columns=['first', 'second', 'first_==_1'])
    #     print(df_X_aft)
    #     print(df_X_exp)
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_not_equal_1(self):
    #     df_X = pd.DataFrame([[1, 2],
    #                          [1, 2],
    #                          [0, 3]],
    #                         columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X,
    #                                                 cont_feature_name='first',
    #                                                 indicator='!=',
    #                                                 number=1)
    #     df_X_exp = pd.DataFrame([[1, 2, False],
    #                              [1, 2, False],
    #                              [0, 3, True]],
    #                             columns=['first', 'second', 'first_!=_1'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_less_2(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='<', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, True], [1, 2, True], [0, 3, True]], columns=['first', 'second', 'first_<_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_greater_2(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='>', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False], [1, 2, False], [0, 3, False]], columns=['first', 'second', 'first_>_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_equal_2(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False], [1, 2, False], [0, 3, False]], columns=['first', 'second', 'first_==_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_not_equal_2(self):
    #     df_X = pd.DataFrame([[1, 2], [1, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, True], [1, 2, True], [0, 3, True]], columns=['first', 'second', 'first_!=_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_equal_1_null(self):
    #     df_X = pd.DataFrame([[1, 2], [np.NaN, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=1)
    #     df_X_exp = pd.DataFrame([[1, 2, False, True], [0.5, 2, True, False], [0, 3, False, False]], columns=['first', 'second', 'first_is_null', 'first_==_1'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_not_equal_1_null(self):
    #     df_X = pd.DataFrame([[1, 2],
    #                          [np.NaN, 2],
    #                          [0, 3]],
    #                         columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=1)
    #     df_X_exp = pd.DataFrame([[1, 2, False, False],
    #                              [0.5, 2, True, True],
    #                              [0, 3, False, True]],
    #                             columns=['first', 'second', 'first_is_null', 'first_!=_1'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_less_2_null(self):
    #     df_X = pd.DataFrame([[1, 2],
    #                          [np.NaN, 2],
    #                          [0, 3]],
    #                         columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='<', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False, True],
    #                              [0.5, 2, True, True],
    #                              [0, 3, False, True]],
    #                             columns=['first', 'second',
    #                                      'first_is_null', 'first_<_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_greater_2_null(self):
    #     df_X = pd.DataFrame([[1, 2], [np.NaN, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='>', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False, False], [0.5, 2, True, False], [0, 3, False, False]], columns=['first', 'second', 'first_is_null', 'first_>_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_equal_2_null(self):
    #     df_X = pd.DataFrame([[1, 2], [np.NaN, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='==', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False, False], [0.5, 2, True, False], [0, 3, False, False]], columns=['first', 'second', 'first_is_null', 'first_==_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())
    #
    #
    # def test_add_feature_continuous_condition_not_equal_2_null(self):
    #     df_X = pd.DataFrame([[1, 2], [np.NaN, 2], [0, 3]], columns=['first', 'second'])
    #     df_X_aft = add_feature_continuous_condition(df_X, cont_feature_name='first', indicator='!=', number=2)
    #     df_X_exp = pd.DataFrame([[1, 2, False, True], [0.5, 2, True, True], [0, 3, False, True]], columns=['first', 'second', 'first_is_null', 'first_!=_2'])
    #     assert_dict_equal(df_X_aft.to_dict(), df_X_exp.to_dict())


    def test_clean_df_X_null(self):
        df_X = pd.DataFrame([[1, 2], [np.NaN, 2], [0, 3]], columns=['first', 'second'])
        print(clean_df_X(df_X))
        df_X_cleaned = pd.DataFrame([[1, 2, False], [.5, 2, True], [0, 3, False]], columns=['first', 'second', 'first_was_null'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())


    def test_clean_df_X_inf(self):
        df_X = pd.DataFrame([[1, 2], [np.inf, 2], [0, 3]], columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 2, False], [.5, 2, True], [0, 3, False]], columns=['first', 'second', 'first_was_inf'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())

    def test_clean_df_X_neg_inf(self):
        df_X = pd.DataFrame([[1, 2], [1, -np.inf], [0, 3]], columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 2, False], [1, 2.5, True], [0, 3, False]], columns=['first', 'second', 'second_was_neg_inf'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())

    def test_clean_df_X_null(self):
        df_X = pd.DataFrame([[1, 2], [1, np.nan], [0, 3]], columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 2, False], [1, 2.5, True], [0, 3, False]], columns=['first', 'second', 'second_was_null'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())

    def test_clean_df_X_neg_null(self):
        df_X = pd.DataFrame([[1, 2], [1, -np.nan], [0, 3]], columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 2, False], [1, 2.5, True], [0, 3, False]], columns=['first', 'second', 'second_was_null'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())

    def test_clean_df_X_null_AND_inf(self):
        df_X = pd.DataFrame([[1, 1], [1, 2], [1, -np.nan], [0, np.inf]], columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 1, False, False], [1, 2, False, False], [1, 1.5, True, False], [0, 1.5, False, True]], columns=['first', 'second', 'second_was_null', 'second_was_inf'])
        assert_dict_equal((clean_df_X(df_X)).to_dict(), df_X_cleaned.to_dict())

    def test_clean_df(self):
        df_X = pd.DataFrame([[1, 1],
                             [1, 2],
                             [1, -np.nan],
                             [0, np.inf]],
                            columns=['first', 'second'])
        df_X_cleaned = pd.DataFrame([[1, 1],
                                     [1, 2]],
                                    columns=['first', 'second'])
        assert_dict_equal(clean_df(df_X, 'second').to_dict(), df_X_cleaned.to_dict())

if __name__ == "__main__":
    unittest.main()
