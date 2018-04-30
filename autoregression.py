import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pandas.tools.plotting import scatter_matrix
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
    plot_partial_depenence,
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
from time import sleep
import sys
import stringcase

# Always make it pretty.
plt.style.use('ggplot')

import os
import importlib.util

# spec = importlib.util.spec_from_file_location("galgraphs", "/Users/macbookpro/Dropbox/Galvanize/autoregression/galgraphs.py")
# galgraphs = importlib.util.module_from_spec(spec)
import galgraphs

# import imp
# galgraphs = imp.load_source('galgraphs', '/Users/macbookpro/Dropbox/Galvanize/autoregression/galgraphs.py')

spec = importlib.util.spec_from_file_location("cleandata", "/Users/macbookpro/Dropbox/Galvanize/autoregression/cleandata.py")
cleandata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleandata)

import tqdm
import time

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

def repeat_spline_pipeliner():
    pass

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

def auto_regression(df, df_test_X, y_var_name, y_test = [], num_alphas=100, alpha_min=.00001, alpha_max=1000000):
    # KEEP ME: FIX BOOLEAN CASE BEFORE DELETING:
    # (continuous_features, category_features) = sort_features(df)
    # df_graphable = df
    # if len(continuous_features)>15:
    #     df_graphable = df[continuous_features[:15]]
    #     print('More continuous features than are graphable in scatter_matrix')
    # pd.scatter_matrix(df_graphable,figsize = (14,len(df_graphable)*.1))
    # plt.show()
    df = data_cleaner.clean_df_respect_to_y(df, y_var_name)
    df_y = df[y_var_name]
    df_X = df.drop(y_var_name, axis = 1)
    df_X = data_cleaner.clean_df_X(df_X)
    df = df_X
    df[y_var_name] = df_y
    num_graphs = int(len(df.columns)/6)
    galgraphs.plot_many_univariates(df, y_var_name)


    # fit model
    (rr_optimized, trained_pipeline, y_cv_mean, y_cv_std) = make_k_folds_ridge(df, y_var_name, num_alphas=num_alphas, alpha_min = alpha_min, alpha_max=alpha_max)
    # apply pipeline to test data
    df_test_X = cleandata.clean_df_X(df_test_X)
    df_test_X_added_features = trained_pipeline.transform(df_test_X)

    #find y_hat
    y_hat = (rr_optimized.predict(df_test_X_added_features) * y_cv_std + y_cv_mean)
    y_hat = make_linear_regression(df, y_var_name, df_test_X)

    #plot coeffs
    galgraphs.plot_coefs(rr_optimized.coef_[0], df_test_X_added_features.columns)

    #plot partial dependencies
    # plot_partial_dependences()

    #plot residuals
    if len(y_test)>0:
        if len(y_test) == len(y_hat):
            (continuous_features, category_features) = sort_features(df_X)
            galgraphs.plot_many_predicteds_vs_actuals(df_X, continuous_features, y_test, y_hat.reshape(-1), n_bins=50)
            fig, ax = plt.subplots()
            galgraphs.plot_residual_error(ax, df_test_X.values[:,0].reshape(-1), y_test.reshape(-1), y_hat.reshape(-1), s=30);
            print(f'MSE = {np.mean((y_hat-y_test)**2)}')
        else:
            print ('len(y_test) != len(y_hat), so no regpressions included' )
    else:
        print( 'No y_test, so no regressions included')
    # (continuous_features, category_features) = sort_features(df)
    # i_s = int(len(continuous_features) / 3)
    # fig, ax = plt.subplots(figsize=( 1, i_s * (len(continuous_features)-i_s) ))
    # for i in range(i_s):
    #     for j in range(i_s,len(continuous_features)):
    #         ax.scatter(df.loc[i], df.loc[j], color="grey")
    #         ax.set_xlabel(df.columns[i])
    #         ax.set_ylabel(df.columns[j])

    # galgraphs.plot_many_predicteds_vs_actuals(df, df.columns, y_var_name, y_hat, n_bins=50)
    return (y_hat, rr_optimized, trained_pipeline, y_cv_mean, y_cv_std)

def compare_predictions(df, y_var_name, percent_data=None, category_limit=11, knots=5, bootstrap_coefs=True, partial_dep=True, actual_vs_predicted=True, residual=True, univariates=True, bootstraps=10):
    def timeit(func, *args):
        start = time.time()
        answers = func(args)

        print(f'{str(func.__name__).upper()} TIME: {time.time() - start}')
        return answers
    df = cleandata.rename_columns(df)
    y_var_name = stringcase.snakecase(y_var_name).replace('__','_')
    start = time.time()
    if percent_data == None:
        while len(df)>1000:
            print(f"'percent_data' NOT SPECIFIED AND len(df)=({len(df)}) IS > 1000: TAKING A RANDOM %10 OF THE SAMPLE")
            df = df.sample(frac=.1)
    else:
        df = df.sample(frac=percent_data)
    print(f'MAKE SUBSAMPLE TIME: {time.time() - start}')

    start = time.time()
    df = cleandata.clean_df(df, y_var_name)
    print(f'CLEAN_DF TIME: {time.time()-start}')

    # REMEMBER OLD DATAFRAME
    df_unpiped = df.copy()
    columns_unpiped = df.columns
    columns_unpiped = list(columns_unpiped)
    columns_unpiped.remove(y_var_name)

    # REMOVE CATEGORICAL VARIABLES THAT HAVE TOO MANY CATEGORIES TO BE USEFUL
    df = cleandata.remove_diverse_categories(df, y_var_name, category_limit)


    # SHOW CORRELATION MATRIX
    if len(df) < 300:
        sample_limit = len(df)
    else:
        sample_limit = 300
    start = time.time()
    plt.matshow(df.sample(sample_limit).corr())
    plt.show()
    print(f'PLOT CORRELATION TIME: {time.time() - start}')

    # MAKE SCATTER MATRIX
    start = time.time()
    galgraphs.plot_scatter_matrix(df, y_var_name)
    plt.show()
    print(f'MAKE SCATTER TIME: {time.time() - start}')
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
    is_continuous = 2 < len(y.unique())
    if is_continuous:
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CONTINUOUS' )
        print()
        if univariates==True:
            galgraphs.plot_many_univariates(df, y_var_name)
            plt.show()
        names_models.append(('LR', LinearRegression())) # LinearRegression as no ._coeff???!
        alphas = np.logspace(start=-5, stop=5, num=5)
        # names_models.append(('RR', RidgeCV(alphas=alphas)))
        names_models.append(('LASSO', LassoCV(alphas=alphas)))
        # names_models.append(('DT', DecisionTreeRegressor()))
        names_models.append(('RF', RandomForestRegressor()))
        # names_models.append(('GB', GradientBoostingRegressor()))
        # names_models.append(('GB', AdaBoostRegressor()))
        # names_models.append(('SVM', SVC()))
        # evaluate each model in turn
        scoring = 'neg_mean_squared_error'
    else:
        alphas = np.logspace(start=-5, stop=5, num=5)
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CATEGORICAL' )
        print()
        names_models.append(('LR', LogisticRegression()))
        # names_models.append(('LDA', LinearDiscriminantAnalysis()))
        # names_models.append(('LC', RidgeClassifierCV(alphas=alphas)))
        # names_models.append(('KNN', KNeighborsClassifier()))
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
        start = time.time()
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        print(model)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        results.append(cv_results)
        names.append(name)
        msg = "%s: mean=%f std=%f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        plt.show()

        print (f'GET CV RESULTS: {time.time()-start}')

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
        start = time.time()
        if is_continuous:
            galgraphs.plot_predicted_vs_actuals(df, model, y_var_name, sample_limit)
            plt.show()

        print(f'PLOT PREDICTED VS ACTUALS TIME: {time.time() - start}')
        # MAKE BOOTSTRAPS
        if bootstrap_coefs or partial_dep:
            bootstrap_models = bootstrap_train_premade(model, X, y, bootstraps=bootstraps, fit_intercept=False)

        #PLOT COEFFICIANTS

        if "coef_" in dir(model):
            start = time.time()
            coefs = model.coef_
            columns=list(df.columns)
            columns.remove(y_var_name)
            while (type(coefs[0]) is list) or (type(coefs[0]) is np.ndarray):
                coefs = list(coefs[0])
            galgraphs.plot_coefs(coefs=coefs, columns=columns, graph_name=name)
            plt.show()

            print(f'PLOT COEFFICIANTS TIME: {time.time() - start}')

            if is_continuous:
                if bootstrap_coefs:
                    # PLOT BOOTSTRAP COEFS
                    start = time.time()
                    fig, axs = plot_bootstrap_coefs(bootstrap_models, df_X.columns, n_col=4)
                    fig.tight_layout()
                    plt.show()

                print(f'PLOT BOOTSTRAP COEFFICIANTS TIME: {time.time() - start}')

        # PLOT PARTIAL DEPENDENCIES
        if partial_dep:
            start = time.time()
            plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            plt.show()
            # galgraphs.shaped_plot_partial_dependences(model, df, y_var_name)
            # plt.show()
            print(f'PLOT PARTIAL DEPENDENCIES TIME: {time.time() - start}')

        # PLOT PREDICTED VS ACTUALS

        df_X_sample = df.sample(sample_limit).drop(y_var_name, axis=1)
        y_hat_sample = model.predict(df_X_sample)
        if is_continuous:
            if len(y)>0:
                if len(y) == len(y_hat_sample):
                    if predicteds_vs_actuals:
                        (continuous_features, category_features) = sort_features(df_X_sample)
                        start = time.time()
                        galgraphs.plot_many_predicteds_vs_actuals(df_X_sample, continuous_features, y, y_hat_sample.reshape(-1), n_bins=50)
                        plt.show()

                        print(f'PLOT PREDICTEDS_VS_ACTUALS TIME: {time.time() - start}')
                        # galgraphs.plot_many_predicteds_vs_actuals(df_X_sample, category_features, y, y_hat_sample.reshape(-1), n_bins=50)
                        # add feature to jitter plot to categorical features
                        # add cdf???
                    if residuals:
                        start = time.time()
                        fig, ax = plt.subplots()
                        galgraphs.plot_residual_error(ax, df_X_sample.values[:,0].reshape(-1), y.reshape(-1), y_hat_sample.reshape(-1), s=30);
                        plt.show()

                    print(f'PLOT RESIDUAL ERROR TIME: {time.time() - start}')
                else:
                    print ('len(y) != len(y_hat), so no regpressions included' )
            else:
                print( 'No y, so no regressions included')

        #Fit MODELS
        df_X = df.drop(y_var_name, axis=1)
        y_hat = model.predict(df_X)

        # get logloss later
        print(f'{name}: MSE = {np.mean((y_hat-y)**2)}')

    # --COMPARE MODELS--
    start = time.time()
    if is_continuous:
        negresults = []
        for i, result in enumerate(results):
            negresults.append(-1*result)
        galgraphs.plot_box_and_violins(names, scoring, negresults)
    else:
        galgraphs.plot_box_and_violins(names, scoring, results)
    plt.show()
    print(f'PLOT BAR AND VIOLIN TIME: {time.time() - start}')
    # ROC CURVE
    # print(f'categorical? {str(!is_continuous)}')
    if not is_continuous:
        start = time.time()
        galgraphs.plot_rocs(models, df_X, y)
        plt.show()

        print(f'PLOT BAR AND VIOLIN TIME: {time.time() - start}')
    return names, results, models, pipeline

def make_one_ridge(df_X_train, y_train, X_test, alpha):
    rr = Ridge(alpha=alpha)
    rr.fit(df_X_train, y_train)
    y_hat = rr.predict(X_test)
    return (y_hat, rr)

def make_ridge_regression(df, y_var_name, pipeliner = auto_spline_pipeliner, knots=10):
    (pipeline, df_X_train, df_X_test, X_train, X_test, y_train, y_test, y_mean, y_std) = cross_var_train(df, y_var_name, pipeliner, knots)
    ridge_regressions = make_ridges(df_X_train, y_train, X_test)
    train_and_test_errors = train_and_test_error(ridge_regressions, df_X_train, y_train, df_X_test, y_test)
    alpha = get_optimal_alpha(train_and_test_errors)
    (y_hat_rr, rr) = make_one_ridge(df_X_train, y_train, X_test, alpha)
    return (y_hat_rr, rr)

def make_plots(df, y_var_name, pipeliner = auto_spline_pipeliner, knots=10):
    (pipeline, df_X_train, df_X_test, X_train, X_test, y_train, y_test, y_mean, y_std) = cross_var_train(df, y_var_name, pipeliner, knots)
    ridge_regressions = make_ridges(df_X_train, y_train, X_test)
    train_and_test_errors = train_and_test_error(ridge_regressions, df_X_train, y_train, df_X_test, y_test)
    alpha = get_optimal_alpha(train_and_test_errors)
    (y_hat_rr, rr) = make_one_ridge(df_X_train, y_train, X_test, alpha)
    (y_hat_lr, lr) = make_linear_regression(df_X_train, y_train, X_test)
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_train_and_test_error(ax, train_and_test_errors)
    plt.show()
    # galgraphs.plot_many_residuals(df_X_test, y_test, y_hat_lr)
    # plt.show()
    # galgraphs.plot_many_residuals(df_X_test, y_test, y_hat_rr)
    # plt.show()

    # Calculate the mean errors across all CV folds, for each regularization strength
    # train_errors = np.empty(shape=(n_folds, len(ridge_regularization_strengths)))
    # for idx, tte in enumerate(errors):
    #     te = tte['train_scores']
    #     train_errors[idx, :] = te
    # mean_train_errors = np.mean(train_errors, axis=0)

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
