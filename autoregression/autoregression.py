import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.stats as scs
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sympy.solvers import solve
from sympy import Symbol
import scipy.optimize as optim
from itertools import product
from sklearn.model_selection import (KFold, train_test_split, cross_val_score)
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline, NaturalCubicSpline)
from regression_tools.plotting_tools import (
    plot_univariate_smooth,
    display_coef,
    bootstrap_train,
    plot_bootstrap_coefs,
    # plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)
from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler, Intercept)
from sklearn import model_selection
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# import warnings
# warnings.filterwarnings('ignore')

import stringcase

# Always make it pretty.
plt.style.use('ggplot')

from autoregression import galgraphs
from autoregression import cleandata
# import galgraphs
# import cleandata

import os
import tqdm
from time import time
import sys

def sort_features(df):
    """Takes a dataframe, returns lists of continuous and category (category) features
    INPUT: dataframe
    OUTPUT: two lists of continuous and category features"""
    continuous_features = []
    category_features = []
    for type, feature in zip(df.dtypes, df.dtypes.index):
        if type == np.dtype('int') or type == np.dtype('float'):
            continuous_features.append(feature)
        if type == np.dtype('O') or type == np.dtype('<U') or type == np.dtype('bool'):
            category_features.append(feature)
    return (continuous_features, category_features)

def auto_spline_pipeliner(df_X, knots=10):
    (continuous_features, category_features) = sort_features(df_X)
    # print(continuous_features)
    # print(category_features)
    continuous_pipelet = []
    category_pipelet = []
    for name in continuous_features:
        knotspace = list(np.linspace(df_X[name].min(), df_X[name].max(), knots))
        continuous_pipelet.append((name+'_fit', galgraphs.simple_spline_specification(name, knotspace)))
    for name in category_features:
        category_pipe = simple_category_specification(name, list(df_X[name].unique()))
        category_pipelet.append((name+'_spec', category_pipe))
        # print(df_X[name].unique()[:-1])
    category_features_pipe = FeatureUnion(category_pipelet)
    if (continuous_features == []) & (category_features == []):
        return "(continuous_features == []) & (category_features == [])"
    if continuous_features == []:
        return category_features_pipe
    continuous_features_scaled = Pipeline([
        ('continuous_features', FeatureUnion(continuous_pipelet)),
        ('standardizer', StandardScaler())
    ])
    if category_features == []:
        return continuous_features_scaled
    pipe_continuous_category = FeatureUnion([
        ('continuous_features', continuous_features_scaled),
        ('category_features', category_features_pipe)
    ])
    return pipe_continuous_category

def is_equal(level):
    def print_equals(var):
        # print('the var is ' + str(var))
        # print('the var is ' + str(level))
        return var==level
    return print_equals

def simple_category_specification(var_name, levels):
    """Make a pipeline taking feature (aka column) 'name' and outputting n-2 new spline features
        INPUT:
            name:
                string, a feature name to spline
            knots:
                int, number knots (divisions) which are divisions between splines.
        OUTPUT:
            pipeline
    """
    select_name = "{}_select".format(var_name)
    map_features = []
    if not isinstance(levels, list):
        levels = [levels]
    for level in levels:
        category_name = "{}_{}_category".format(var_name, level)
        map_features.append(
            (category_name, MapFeature(is_equal(level), category_name))
        )
        # print('The var is "' + str(var_name))
        # print('The level is  "' + str(level))
    return Pipeline([
        (select_name, ColumnSelector(name=var_name)),
        ("category_features", FeatureUnion(map_features))
    ])

def timeit(func, *args):
    start = time()
    answers = func(args)

    print(f'{str(func.__name__).upper()} TIME: {time() - start}')
    return answers


def compare_predictions(df, y_var_name, percent_data=None,
                        category_limit=11, knots=3, bootstrap_coefs=True,
                        partial_dep=True, actual_vs_predicted=True,
                        residuals=True, univariates=True, bootstraps=10):
    df = cleandata.rename_columns(df)
    y_var_name = stringcase.snakecase(y_var_name).replace('__', '_')
    start = time()
    if percent_data is None:
        while len(df) > 1000:
            print(f"""'percent_data' NOT SPECIFIED AND len(df)=({len(df)})
                  IS > 1000: TAKING A RANDOM %10 OF THE SAMPLE""")
            df = df.sample(frac=.1)
    else:
        df = df.sample(frac=percent_data)
    print(f'MAKE SUBSAMPLE TIME: {time() - start}')

    start = time()
    df = cleandata.clean_df(df, y_var_name)
    print(f'CLEAN_DF TIME: {time()-start}')

    # REMEMBER OLD DATAFRAME

    df_unpiped = df.copy()
    (unpiped_continuous_features, unpiped_category_features) = sort_features(df_unpiped.drop(y_var_name, axis=1))
    columns_unpiped = df.columns
    columns_unpiped = list(columns_unpiped)
    columns_unpiped.remove(y_var_name)

    # REMOVE CATEGORICAL VARIABLES THAT HAVE TOO MANY CATEGORIES TO BE USEFUL
    df = cleandata.drop_category_exeeding_limit(df, y_var_name, category_limit)


    # SHOW CORRELATION MATRIX
    if len(df) < 300:
        sample_limit = len(df)
    else:
        sample_limit = 300
    start = time()
    plt.matshow(df.sample(sample_limit).corr())
    plt.show()
    print(f'PLOT CORRELATION TIME: {time() - start}')

    # MAKE SCATTER MATRIX
    start = time()
    galgraphs.plot_scatter_matrix(df, y_var_name)
    plt.show()
    print(f'MAKE SCATTER TIME: {time() - start}')
    print()


    print('DF COLUMNS: ')
    print(str(list(df.columns)))
    print()
    # TRANSFORM DATAFRAME
    df_X = df.drop(y_var_name, axis = 1)
    pipeline = auto_spline_pipeliner(df_X, knots=5)
    pipeline.fit(df_X)
    df_X = pipeline.transform(df_X)
    X = df_X.values
    y = df[y_var_name]
    df = df_X
    df[y_var_name] = y
    print('DF COLUMNS AFTER TRANSFORM: ')
    print(str(list(df.columns)))
    print()

    # CHOOSE MODELS FOR CONTINUOUS OR CATEGORICAL Y
    names_models = []
    print(len(y.unique()))

    (continuous_features, category_features) = sort_features(df_X)
    is_continuous = (y_var_name in continuous_features)
    if is_continuous:
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CONTINUOUS' )
        print()
        if univariates==True:
            galgraphs.plot_many_univariates(df, y_var_name)
            plt.show()
        # names_models.append(('LR', LinearRegression()))
        alphas = np.logspace(start=-2, stop=5, num=5)
        names_models.append(('RR', RidgeCV(alphas=alphas)))
        names_models.append(('LASSO', LassoCV(alphas=alphas)))
        names_models.append(('DT', DecisionTreeRegressor()))
        names_models.append(('RF', RandomForestRegressor()))
        names_models.append(('GB', GradientBoostingRegressor()))
        names_models.append(('GB', AdaBoostRegressor()))
        # names_models.append(('SVM', SVC()))
        # evaluate each model in turn
        scoring = 'neg_mean_squared_error'
    else:
        alphas = np.logspace(start=-5, stop=5, num=5)
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CATEGORICAL' )
        print()
        names_models.append(('LR', LogisticRegression()))
        names_models.append(('LDA', LinearDiscriminantAnalysis()))
        names_models.append(('LC', RidgeClassifierCV(alphas=alphas)))
        names_models.append(('KNN', KNeighborsClassifier()))
        names_models.append(('DT', DecisionTreeClassifier()))
        # names_models.append(('NB', GaussianNB()))
        names_models.append(('RF', RandomForestClassifier()))
        names_models.append(('GB', GradientBoostingClassifier()))
        names_models.append(('DT', AdaBoostClassifier()))
        # names_models.append(('SVM', SVC()))
        scoring = 'accuracy'
    models = [x[1] for x in names_models]
    fit_models = []

    # evaluate each model in turn
    results = []
    names = []
    seed = 7
    for name, model in tqdm.tqdm(names_models):

        # CROSS VALIDATE MODELS
        start = time()
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        results.append(cv_results)
        names.append(name)
        msg = "%s: mean=%f std=%f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        plt.show()

        print (f'CV CALC TIME: {time()-start}')

        # #OTHER CROSS VALIDATE METHOD:
        # ridge_regularization_strengths = np.logspace(np.log10(0.000001), np.log10(100000000), num=100)
        # ridge_regressions = []
        # y=df['age']
        # df_X = df.drop('age', axis=1)
        # for alpha in ridge_regularization_strengths:
        #     ridge = Ridge(alpha=alpha)
        #     ridge.fit(df_X, y)
        #     ridge_regressions.append(ridge)
        # fig, ax = plt.subplots(figsize=(16, 6))
        # galgraphs.plot_solution_paths(ax, ridge_regressions)


        # ADD GRIDSEARCH HERE

        # FIT MODEL WITH ALL DATA
        model.fit(X,y)
        fit_models.append(model)

        # PLOT PREDICTED VS ACTUALS
        start = time()
        if is_continuous:
            galgraphs.plot_predicted_vs_actuals(df, model, y_var_name, sample_limit)
            plt.show()

        print(f'PLOT PREDICTED VS ACTUALS TIME: {time() - start}')
        # MAKE BOOTSTRAPS
        if bootstrap_coefs or partial_dep:
            bootstrap_models = bootstrap_train_premade(model, X, y, bootstraps=bootstraps, fit_intercept=False)

        #PLOT COEFFICIANTS

        if hasattr(model, "coef_"):
            start = time()
            coefs = model.coef_
            columns = list(df.columns)
            columns.remove(y_var_name)
            while (type(coefs[0]) is list) or (type(coefs[0]) is np.ndarray):
                coefs = list(coefs[0])
            galgraphs.plot_coefs(coefs=coefs, columns=columns, graph_name=name)
            plt.show()

            print(f'PLOT COEFFICIANTS TIME: {time() - start}')

            if is_continuous:
                if bootstrap_coefs:
                    # PLOT BOOTSTRAP COEFS
                    start = time()
                    fig, axs = plot_bootstrap_coefs(bootstrap_models, df_X.columns, n_col=4)
                    fig.tight_layout()
                    plt.show()

                print(f'PLOT BOOTSTRAP COEFFICIANTS TIME: {time() - start}')

        # PLOT PARTIAL DEPENDENCIES
        if partial_dep:
            start = time()
            plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=unpiped_continuous_features, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            # plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            # galgraphs.plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            plt.show()
            print(f'PLOT CONTINOUS PARTIAL DEPENDENCIES TIME: {time() - start}')
            start = time()
            hot_categorical_vars = [column for column in df.columns if (len(df[column].unique()) == 2)]
            # galgraphs.shaped_plot_partial_dependences(model, df[[y_var_name]+hot_categorical_vars], y_var_name)
            plt.show()
            print(f'PLOT CATEGORICAL PARTIAL DEPENDENCIES TIME: {time() - start}')

        # PLOT PREDICTED VS ACTUALS

        df_X_sample = df.sample(sample_limit).drop(y_var_name, axis=1)
        y_hat_sample = model.predict(df_X_sample)
        if is_continuous:
            if len(y)>0:
                if len(y) == len(y_hat_sample):
                    if predicteds_vs_actuals:
                        (continuous_features, category_features) = sort_features(df_X_sample)
                        start = time()
                        galgraphs.plot_many_predicteds_vs_actuals(df_X_sample, continuous_features, y, y_hat_sample.reshape(-1), n_bins=50)
                        plt.show()

                        print(f'PLOT PREDICTEDS_VS_ACTUALS TIME: {time() - start}')
                        # galgraphs.plot_many_predicteds_vs_actuals(df_X_sample, category_features, y, y_hat_sample.reshape(-1), n_bins=50)
                        # add feature to jitter plot to categorical features
                        # add cdf???
                    if residuals:
                        start = time()
                        fig, ax = plt.subplots()
                        galgraphs.plot_residual_error(ax, df_X_sample.values[:,0].reshape(-1), y.reshape(-1), y_hat_sample.reshape(-1), s=30);
                        plt.show()

                    print(f'PLOT RESIDUAL ERROR TIME: {time() - start}')
                else:
                    print ('len(y) != len(y_hat), so no regressions included' )
            else:
                print( 'No y, so no regressions included')

        #Fit MODELS
        df_X = df.drop(y_var_name, axis=1)
        y_hat = model.predict(df_X)

        # get logloss later
        print(f'{name}: MSE = {np.mean((y_hat-y)**2)}')

    # --COMPARE MODELS--
    start = time()
    if is_continuous:
        negresults = []
        for i, result in enumerate(results):
            negresults.append(-1*result)
        galgraphs.plot_box_and_violins(names, scoring, negresults)
    else:
        galgraphs.plot_box_and_violins(names, scoring, results)
    plt.show()
    print(f'PLOT BAR AND VIOLIN TIME: {time() - start}')
    # ROC CURVE
    if not is_continuous:
        start = time()
        galgraphs.plot_rocs(models, df_X, y)
        plt.show()
        print(f'PLOT ROC TIME: {time() - start}')
    return names, results, models, pipeline

def bootstrap_train_premade(model, X, y, bootstraps=1000, **kwargs):
    """Train a (linear) model on multiple bootstrap samples of some data and
    return all of the parameter estimates.

    Parameters
    ----------
    model: A sklearn class whose instances have a `fit` method, and a `coef_`
    attribute.

    X: A two dimensional numpy array of shape (n_observations, n_features).

    y: A one dimensional numpy array of shape (n_observations).

    bootstraps: An integer, the number of boostrapped names_models to train.

    Returns
    -------
    bootstrap_coefs: A (bootstraps, n_features) numpy array.  Each row contains
    the parameter estimates for one trained boostrapped model.
    """
    bootstrap_models = []
    for i in range(bootstraps):
        boot_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[boot_idxs, :]
        y_boot = y[boot_idxs]
        model.fit(X_boot, y_boot)
        bootstrap_models.append(model)
    return bootstrap_models

if __name__ == "__main__":
    balance_non_zero = pd.read_csv("/Users/macbookpro/Dropbox/Galvanize/lectures/DSI_Lectures/regularized-regression/matt_drury/balance_non_zero.csv", index_col=0)
    balance = balance_non_zero.head(1000)
    # balance_with_zero = pd.read_csv("/Users/macbookpro/Dropbox/Galvanize/dsi-practical-linear-regression/data/balance.csv", index_col=0)
    # balance = balance_with_zero.head(10000)

    # make_plots(balance, 'Balance', auto_spline_pipeliner, 10)
    (rr_optimized, auto_spline_pipeline) = make_k_folds_ridge(balance, 'Balance', pipeliner = auto_spline_pipeliner, knots = 10, alpha_min = 10, alpha_max = 1000)
    # import pdb;
    # pdb.set_trace()
    # print(rr_optimized.predict(auto_spline_pipeline.transform(balance).values))
