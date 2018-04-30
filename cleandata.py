import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scs
import operator
import galgraphs
import importlib.util
import stringcase
spec = importlib.util.spec_from_file_location("galgraphs", "/Users/macbookpro/Dropbox/Galvanize/galgraphs/galgraphs/galgraphs.py")
galgraphs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(galgraphs)

import sys 
import os
sys.path.append(os.path.abspath("/Users/macbookpro/Dropbox/Galvanize/autoregression/"))
import autoregression

def rename_columns(df):
    replacers = {}
    for name in df.columns:
        replacers[name] = stringcase.snakecase(name).replace('__','_')
    df = df.rename(index=str, columns=replacers)
    return df

def add_feature_continuous_condition(df_X, feature_name, indicator, number):
    ops = {"==": operator.is_, 
           "!=": operator.is_not, 
           '<': operator.lt,
           '<=': operator.le,
          '>': operator.gt,
          '>=': operator.ge}
    if pd.isnull(df_X[feature_name]).any():
        df_X[feature_name + "_is_null"] = pd.isnull(df_X[feature_name])
        df_X[feature_name][pd.isnull(df_X[feature_name])] = np.mean(df_X[feature_name][~pd.isnull(df_X[feature_name])])
#     df_X[feature_name + "_is_na"] = pd.isna(df_X[feature_name])
#     df_X[feature_name][~pd.isna(df_X[feature_name])] = np.mean(df_X[feature_name][~pd.isna(df_X[feature_name])])
    df_X[feature_name + "_" + str(indicator) + "_" + str(number)] = ops[indicator](df_X[feature_name], number)
    df_X[feature_name][ops[indicator](df_X[feature_name],number)] = np.mean(df_X[feature_name][~ops[indicator](df_X[feature_name],number)])
    return df_X

def add_feature_continuous_null(df_X, feature_name):
    if (df_X[feature_name] == np.inf).any():
        df_X[feature_name + "_was_inf"] = (df_X[feature_name] == np.inf)
        df_X[feature_name][df_X[feature_name] == np.inf] = np.mean(df_X[feature_name][df_X[feature_name] != np.inf])

    if (df_X[feature_name] == -np.inf).any():
        df_X[feature_name + "_was_neg_inf"] = (df_X[feature_name] == np.inf)
        df_X[feature_name][df_X[feature_name] == -np.inf] = np.mean(df_X[feature_name][df_X[feature_name] != -np.inf])
        
    if pd.isnull(df_X[feature_name]).any():
        df_X[feature_name + "_was_null"] = pd.isnull(df_X[feature_name])
        df_X[feature_name][pd.isnull(df_X[feature_name])] = np.mean(df_X[feature_name][~pd.isnull(df_X[feature_name])])
    return df_X

def category_clean_null_and_inf(df_X, feature_name):
    df_X[feature_name][df_X[feature_name] == np.inf] = "was_inf"
    df_X[feature_name][df_X[feature_name] == -np.inf] = "was_neg_inf"
    df_X[feature_name][pd.isnull(df_X[feature_name])] = "was_null"
    return df_X

def clean_df_X(df_X):
    (continuous_features, categorical_features) = autoregression.sort_features(df_X)
    for feature in continuous_features:
        df_X = add_feature_continuous_null(df_X, feature)
    for feature in categorical_features:
        df_X = category_clean_null_and_inf(df_X, feature)
    for feature in continuous_features:
        if (len(df_X[feature].unique()) <= 1):
            df_X = df_X.drop(feature, axis=1)
    return df_X

def clean_df_respect_to_y(df, y_var_name):
    return df[~df[y_var_name].isnull()]

def clean_df(df, y_var_name):
    df = clean_df_respect_to_y(df, y_var_name)
    df_y = df[y_var_name]
    df_X = df.drop(y_var_name, axis = 1)
    df_X = clean_df_X(df_X)
    df = df_X
    df[y_var_name] = df_y
    return df
# TODO: Turn these into pipelines