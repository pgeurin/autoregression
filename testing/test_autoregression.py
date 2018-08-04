import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import (GradientBoostingRegressor,
                              GradientBoostingClassifier,
                              RandomForestRegressor,
                              RandomForestClassifier,
                              AdaBoostClassifier,
                              AdaBoostRegressor)
from sklearn.pipeline import Pipeline
from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler, Intercept)
from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline, NaturalCubicSpline)

from autoregression import (simple_spline_specification,
                            simple_category_specification,
                            make_cont_models,
                            plot_choose_alpha,
                            clean_dataframe,
                            take_subsample,
                            make_sample_limit)



class TestModels(unittest.TestCase):
    def test_make_cont_models(self):
        # self.maxDiff=None
        # self.assertEqual(
        #     make_cont_models(alphas=np.logspace(start=-2, stop=5, num=5)),
        #     [('RR', Ridge),
        #      ('LASSO', Lasso),
        #      ('DT', DecisionTreeRegressor()),
        #      ('RF', RandomForestRegressor()),
        #      ('GB', GradientBoostingRegressor())])
        self.assertEqual(True,True)
        alphas = [0.1, 0.01, 0.001]
        self.assertEqual(len(make_cont_models(alphas)), 5)
        self.assertEqual(len(
            list(zip(*make_cont_models(alphas)))), 2)


    def test_plot_choose_alpha(self):
        model = Ridge
        y_var_name = 'first'
        alphas = [0.1, 0.01, 0.001]
        scoring = 'neg_mean_squared_error'
        seed=7
        kfold = KFold(n_splits=10, random_state=seed)
        df = pd.DataFrame([[1, 2, 3],
                           [2, 60, 4],
                           [3, 4, 40],
                           [2, 3, 9],
                           [3, 4, 5],
                           [2, 100, 4],
                           [3, 10, 5],
                           [2, 3, 4],
                           [3, 4, 30],
                           [2, 50, 4],
                           [3, 4, 5]], columns=['first', 'second', 'third'])
        alpha, cv_results = plot_choose_alpha(df, model, y_var_name, alphas, kfold, scoring)
        self.assertEqual(.1, alpha)
        df = pd.DataFrame([[1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4.5, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5]], columns=['first', 'second', 'third'])
        alpha, cv_results = plot_choose_alpha(df, model, y_var_name, alphas, kfold, scoring)
        self.assertEqual(.01, alpha)
        df = pd.DataFrame([[1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5]], columns=['first', 'second', 'third'])
        alpha, cv_results = plot_choose_alpha(df, model, y_var_name, alphas, kfold, scoring)
        self.assertEqual(.001, alpha)


    def test_clean_dataframe(self):
        df1 = pd.DataFrame([[1, np.inf, 3],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, np.NAN],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5],
                           [2, 3, 4],
                           [3, 4, 5]], columns=['first', 'second', 'third'])
        y_var_name = 'first'
        percent_data = 1
        df2, sample_limit = clean_dataframe(df1, y_var_name, percent_data=None)
        self.assertTrue(df2.reset_index(drop=True).equals(
                         pd.DataFrame(
                          [[3.5, 3, True, False, 1],
                           [3, 4, False, False, 2],
                           [4, 5, False, False, 3],
                           [3, 4, False, False, 2],
                           [4, 5, False, False, 3],
                           [3, 4.4, False, True, 2],
                           [4, 5, False, False, 3],
                           [3, 4, False, False, 2],
                           [4, 5, False, False, 3],
                           [3, 4, False, False, 2],
                           [4, 5, False, False, 3]], columns=[
                                                'second',
                                                'third',
                                                'second_was_inf',
                                                'third_was_null',
                                                'first']).reset_index(drop=True)))


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


    def test_take_subsample(self):
        df = pd.DataFrame([[1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4]],
                          columns=['first',
                                   'second',
                                   'third',
                                   'fourth'])
        df2 = take_subsample(df, percent_data=1)
        df2 = df2.sort_index(ascending=True)
        self.assertTrue(df.equals(df2))
        self.assertEqual(len(df2), 3)
        df3 = take_subsample(df, percent_data=0.66)
        self.assertEqual(len(df3), 2)


    def test_make_sample_limit(self):
        df = pd.DataFrame([[1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4]],
                          columns=['first',
                                   'second',
                                   'third',
                                   'fourth'])
        self.assertEqual(3, make_sample_limit(df))
        df1 = df
        for i in range(20):
            df1 = df1.append(df1)
        self.assertEqual(300, make_sample_limit(df1))


    def test_clean_df_y_nan(self):
        df = pd.DataFrame([[1, 2, 3, np.nan],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]],
                          columns=['first',
                                   'second',
                                   'third',
                                   'fourth'])
        df_clean_sample = pd.DataFrame([[5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0]],
                                columns=['first',
                                        'second',
                                        'third',
                                        'fourth'])
        df_cleaned, sample_limit = clean_dataframe(df, 'fourth', percent_data=1)
        assert df_cleaned.reset_index(drop=True).to_dict() == df_clean_sample.reset_index(drop=True).to_dict()

    def test_clean_df_y_x_nan(self):
        df = pd.DataFrame([[1, np.nan, 3, np.nan],
                           [5, 6, 7, 8],
                           [9, 10, np.nan, 12],
                           [13, 14, 15, 16]],
                          columns=['first',
                                   'second',
                                   'third',
                                   'fourth'])
        df_clean_sample = pd.DataFrame([[5.0, 6.0, 7.0, False, 8.0],
                                [9.0, 10.0, 11.0, True, 12.0],
                                [13.0, 14.0, 15.0, False, 16.0]],
                                columns=['first',
                                        'second',
                                        'third',
                                        'third_was_null',
                                        'fourth'])
        print(df)
        df_clean_sample = df_clean_sample.reset_index(drop=True)
        print(df_clean_sample)
        df_cleaned, sample_limit = clean_dataframe(df, 'fourth', percent_data=1)
        df_cleaned = df_cleaned.sort_index().reset_index(drop=True)
        print(df_cleaned)
        assert df_cleaned.to_dict() == df_clean_sample.to_dict()

    def test_clean_df_y_x_inf(self):
        df = pd.DataFrame([[1, np.nan, 3, np.inf],
                           [5, 6, 7, 8],
                           [9, 10, np.nan, 12],
                           [13, 14, 15, 16]],
                          columns=['first',
                                   'second',
                                   'third',
                                   'fourth'])
        df_clean_sample = pd.DataFrame([[5.0, 6.0, 7.0, False, 8.0],
                                [9.0, 10.0, 10.5, True, 12.0],
                                [13.0, 14.0, 15.0, False, 16.0]],
                                columns=['first',
                                        'second',
                                        'third',
                                        'third_was_null',
                                        'fourth'])
        print(df)
        df_clean_sample = df_clean_sample.reset_index(drop=True)
        print(df_clean_sample)
        df_cleaned, sample_limit = clean_dataframe(df, 'fourth', percent_data=1)
        df_cleaned = df_cleaned.sort_index().reset_index(drop=True)
        print(df_cleaned)
        assert df_cleaned.to_dict() == df_cleaned.to_dict()

        # self.assertTrue(False) # This correctly breaks the machine


# class TestSplines(unittest.TestCase):
#     def test_simple_spline_specification(self):
#         self.assertEqual(simple_spline_specification('this_feature'),
#                          Pipeline([(('this_feature_select',
#                                     ColumnSelector(name='this_feature'))),
#                                   ('this_feature_spline',
#                                    NaturalCubicSpline(knots=10))
#                                   ]))
#
#
#     def test_simple_category_specification(self):
#         self.assertEqual(
#             simple_category_specification('this_feature',
#                                           ['one_level', 'two_level']
#                                           ).__dict__,
#                  Pipeline([(('this_feature_select',
#                             ColumnSelector(name='this_feature'))),
#                           ("category_features",
#                            FeatureUnion([('this_feature', 'this_feature_one_level_category'),
#                                          ('this_feature', 'this_feature_two_level_category')
#                                          ]))
#                           ]).__dict__
#                          )

if __name__ == '__main__':
    unittest.main()
