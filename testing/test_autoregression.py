import unittest
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
                            make_cont_models)



class TestStringMethods(unittest.TestCase):
    def test_make_cont_models(self):
        self.maxDiff=None
        self.assertEqual(
            make_cont_models(alphas=np.logspace(start=-2, stop=5, num=5)),
                                        [('RR', Ridge),
                                         ('LASSO', Lasso),
                                         ('DT', DecisionTreeRegressor()),
                                         ('RF', RandomForestRegressor()),
                                         ('GB', GradientBoostingRegressor())])
        self.assertEqual(True,True)
        self.assertEqual(len(make_cont_models(alphas)),5)
        self.assertEqual(len(zip(make_cont_models(alphas))),2)


    def test_simple_spline_specification(self):
        self.assertEqual(simple_spline_specification('this_feature'),
                         Pipeline([(('this_feature_select',
                                    ColumnSelector(name='this_feature'))),
                                  ('this_feature_spline',
                                   NaturalCubicSpline(knots=10))
                                  ])
                         )


    def test_simple_category_specification(self):
        self.assertEqual(simple_category_specification('this_feature', ['one_level', 'two_level']),
                         Pipeline([(('this_feature_select',
                                    ColumnSelector(name='this_feature'))),
                                  ("category_features",
                                   FeatureUnion([('this_feature', 'this_feature_one_level_category'),
                                                 ('this_feature', 'this_feature_two_level_category')
                                                 ]))
                                  ])
                         )


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
