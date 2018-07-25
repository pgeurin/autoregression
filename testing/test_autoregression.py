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
                            clean_dataframe)



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


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
