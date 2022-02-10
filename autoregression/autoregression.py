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
from sklearn.model_selection import (KFold,
                                     train_test_split,
                                     cross_val_score)
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn import model_selection
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import Ridge, Lasso, RidgeClassifier, LogisticRegression, RidgeCV, LassoCV, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
# import warnings
# warnings.filterwarnings('ignore')
import stringcase
from autoregression import cleandata
from autoregression.cleandata import (sort_features,
                                      rename_columns,
                                      clean_df,
                                      drop_category_exeeding_limit)
from autoregression import galgraphs
from autoregression.galgraphs import (plot_many_univariates,
                                      plot_scatter_matrix,
                                      plot_solution_paths,
                                      plot_predicted_vs_actuals,
                                      plot_coefs,
                                      # plot_partial_dependences
                                      plot_feature_importances,
                                      plot_many_predicteds_vs_actuals,
                                      plot_residual_error,
                                      plot_box_and_violins,
                                      plot_rocs)
import os
import tqdm
from time import time
import sys
# Always make it pretty.
plt.style.use('ggplot')


def choose_alpha(df, model, y_var_name, alphas, kfold, scoring, plot_alphas=False):
    ridges = []
    ridges_scores = []
    ave_scores = []
    y = df[y_var_name]
    df_X = df.drop(y_var_name, axis=1)
    for alpha in alphas:
        ridge = model(alpha=alpha)
        ridge.fit(df_X, y)
        ridges.append(ridge)
        ridge_scores = cross_val_score(ridge, df_X.values, y, cv=kfold,
                                       scoring=scoring)
        ridges_scores.append(ridge_scores)
        ave_score = np.mean(ridge_scores)
        ave_scores.append(ave_score)
    if plot_alphas:
        fig, ax = plt.subplots(figsize=(16, 6))
        plot_solution_paths(ax, ridges)
    best_index = np.argmax(ave_scores)   # IS THIS MIN OR MAX???
    return alphas[best_index], ridges_scores[best_index]


def plot_continuous_error_graphs(df, y, y_var_name, model,
                                 is_continuous, sample_limit=300,
                                 predicteds_vs_actuals=True,
                                 residuals=True):
    df_X_sample = df.sample(sample_limit).drop(y_var_name, axis=1)
    y_hat_sample = model.predict(df_X_sample)
    if is_continuous:
        if len(y) > 0:
            if len(y) == len(y_hat_sample):
                if predicteds_vs_actuals:
                    (continuous_features,
                     category_features) = sort_features(df_X_sample)
                    timeit(plot_many_predicteds_vs_actuals, df_X_sample,
                           continuous_features, y, y_hat_sample.reshape(-1),
                           n_bins=50)
                    plt.draw()
                    # add feature to jitter plot to categorical features
                    # add cdf???
                if residuals:
                    fig, ax = plt.subplots()
                    print(y)
                    timeit(plot_residual_error, ax,
                           df_X_sample.values[:, 0].reshape(-1),
                           y, y_hat_sample, s=30) # had .reshape on y and y_hat_sample. Removed for fix?
                    plt.draw()
            else:
                print('len(y) != len(y_hat), so no regressions included')
        else:
            print('No y, so no regressions included')
    return None


def choose_box_and_violin_plots(names, scoring, compare_models,
                                results, is_continuous):
    if is_continuous:
        negresults = []
        for i, result in enumerate(results):
            negresults.append(-1*result)
        timeit(plot_box_and_violins, names, scoring, negresults)
    else:
        timeit(plot_box_and_violins, names, scoring, results)
    plt.draw()
    return None


def auto_spline_pipeliner(df_X, knots=10):
    (continuous_features, category_features) = sort_features(df_X)
    # print(continuous_features)
    # print(category_features)
    continuous_pipelet = []
    category_pipelet = []
    for name in continuous_features:
        knotspace = list(np.linspace(df_X[name].min(),
                                     df_X[name].max(),
                                     knots))
        continuous_pipelet.append((name+'_fit',
                                   simple_spline_specification(name,
                                                               knotspace)))
    for name in category_features:
        category_pipe = simple_category_specification(name,
                                                      list(df_X[name].unique())
                                                      )
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
        return var == level
    return print_equals


def simple_spline_specification(name, knots=10):
    """Make a pipeline taking feature (aka column) 'name' and outputting n-2
    new spline features
        INPUT:
            name:
                string, a feature name to spline
            knots:
                int, number knots (divisions) which are divisions between
                splines.
        OUTPUT:
            pipeline returning of n-2 new splines after transformed
    """
    select_name = "{}_select".format(name)
    spline_name = "{}_spline".format(name)
    return Pipeline([
        (select_name, ColumnSelector(name=name)),
        (spline_name, NaturalCubicSpline(knots=knots))
    ])


def simple_category_specification(var_name, levels):
    """Make a pipeline taking feature (aka column) 'name' and
        outputting n-2 new levels
        INPUT:
            name:
                string, a feature name to spline
            levels:
                int, number knots (divisions) which are divisions
                between splines.
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


def timeit(func, *args, **kwargs):
    print(f'Running: {func.__name__.upper()} ...')
    start = time()
    answers = func(*args, **kwargs)
    print(f'{func.__name__.upper()} TIME: {time() - start}')
    return answers


def make_cont_models(alphas):
    names_models = []
    # names_models.append(('LR', LinearRegression()))
    names_models.append(('RR', Ridge))
    names_models.append(('LASSO', Lasso))
    names_models.append(('DT', DecisionTreeRegressor()))
    names_models.append(('RF', RandomForestRegressor()))
    names_models.append(('GB', GradientBoostingRegressor()))
    # names_models.append(('GB', AdaBoostRegressor()))
    # names_models.append(('SVM', SVC()))
    return names_models

def make_cat_models(alphas):
    names_models = []
    names_models.append(('LR', LogisticRegression()))
    # names_models.append(('LASSOish', LogisticRegression(penalty='l1')))
    # names_models.append(('LDA', LinearDiscriminantAnalysis()))
    names_models.append(('RR', RidgeClassifier))
    names_models.append(('KNN', KNeighborsClassifier()))
    names_models.append(('DT', DecisionTreeClassifier()))
    # names_models.append(('NB', GaussianNB()))
    names_models.append(('RF', RandomForestClassifier()))
    names_models.append(('GB', GradientBoostingClassifier()))
    # names_models.append(('DT', AdaBoostClassifier()))
    # names_models.append(('SVM', SVC()))
    return names_models

def make_models(df, df_X, y, y_var_name, univariates,
                alphas=np.logspace(start=-2, stop=5, num=5)):
    """CHOOSE MODELS FOR CONTINUOUS OR CATEGORICAL Y, make the Models"""
    print(len(y.unique()))
    (continuous_features, category_features) = sort_features(df_X)
    is_continuous = (y_var_name in continuous_features)
    if is_continuous:
        print('Y VARIABLE: "' + y_var_name + '" IS CONTINUOUS')
        print()
        if univariates:
            plot_many_univariates(df, y_var_name)
            plt.draw()
        names_models = make_cont_models(alphas)
        scoring = 'neg_mean_squared_error'
    else:
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CATEGORICAL' )
        print()
        names_models = make_cat_models(alphas)
        scoring = 'roc_auc'
    models = [x[1] for x in names_models]
    return (names_models, continuous_features, category_features,
            models, scoring, is_continuous, alphas)


def take_subsample(df, percent_data=None):
    if percent_data is None:
        while len(df) > 1000:
            print(f"""'percent_data' NOT SPECIFIED AND len(df)=({len(df)})
                  IS > 1000: TAKING A RANDOM %10 OF THE SAMPLE""")
            df = df.sample(frac=.1)
    else:
        df = df.sample(frac=percent_data)
    return df


def make_sample_limit(df):
    if len(df) < 300:
        sample_limit = len(df)
    else:
        sample_limit = 300
    return sample_limit


def use_spline(df, y_var_name):
    df_X = df.drop(y_var_name, axis=1)
    pipeline = auto_spline_pipeliner(df_X, knots=5)
    pipeline.fit(df_X)
    df_X = pipeline.transform(df_X)
    X = df_X.values
    y = df[y_var_name]
    df = df_X
    df[y_var_name] = y
    return df, df_X, X, y, pipeline


def get_error(name, model, df_X, y, is_continuous):
    if is_continuous:
        y_hat = model.predict(df_X)
        mse = np.mean((y_hat-y)**2)
        print(f'{name}: MSE = {mse}')
        error = mse
    else:
        if 'predict_proba' in dir(model):
            y_hat = model.predict_proba(df_X)[:, 0]
            # print(model)
            # print(y)
            # print(y_hat)
            logloss = np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            print(f'{name}: logloss = {logloss}')
            error = logloss
        if 'decision_function' in dir(model):
            d = model.decision_function(df_X)[0]
            y_hat = np.exp(d) / np.sum(np.exp(d))
            mse = np.mean((y_hat-y)**2)
            print(f'{name}: logloss = {mse}')
            error = mse
    return y_hat, error


def clean_dataframe(df, y_var_name, percent_data):
        df = rename_columns(df)
        y_var_name = stringcase.snakecase(y_var_name).replace('__',
                                                              '_'
                                                              ).replace('__',
                                                                        '_')
        df = timeit(take_subsample, df, percent_data)
        df = timeit(clean_df, df, y_var_name)
        sample_limit = make_sample_limit(df)
        return df, sample_limit


def compare_predictions(df, y_var_name, percent_data=None,
                        category_limit=11, knots=3,
                        alphas=np.logspace(start=-2, stop=10, num=50),
                        corr_matrix=False,
                        scatter_matrix=False, 
                        bootstrap_coefs=False,
                        partial_dep=False, 
                        plot_alphas=False,
                        plot_predicted_vs_actuals=False,
                        plot_coefs_flag=False,
                        feature_importances=False,
                        actual_vs_predicted=False,
                        plot_predicteds_vs_actuals=False,
                        residuals=False, 
                        univariates=False, 
                        compare_models=False,
                        ROC=False, 
                        bootstraps=10):
    """Takes dataframe
        INPUT:
            name:
                string, a feature name to spline
            knots:
                int, number knots (divisions) which are
                divisions between splines.
        OUTPUT:
            pipeline
    """
    starttotal = time()
    df, sample_limit = clean_dataframe(df, y_var_name, percent_data)

    # REMEMBER OLD DATAFRAME

    df_unpiped, df_X_unpiped = df.copy(), df.copy().drop(y_var_name, axis=1)
    (unpiped_continuous_features,
     unpiped_category_features) = sort_features(df_X_unpiped)
    columns_unpiped = df_X_unpiped.columns

    # REMOVE CATEGORICAL VARIABLES THAT HAVE TOO MANY CATEGORIES TO BE USEFUL
    df = drop_category_exeeding_limit(df, y_var_name, category_limit)

    # SHOW CORRELATION MATRIX
    if corr_matrix:
        if len(unpiped_continuous_features) > 0:
            timeit(plt.matshow, df.sample(sample_limit).corr())
            plt.draw()

    # TRANSFORM DATAFRAME
    print('DF COLUMNS: \n' + str(list(df.columns)) + '\n')
    df, df_X, X, y, pipeline = use_spline(df, y_var_name)
    print('DF COLUMNS AFTER TRANSFORM: \n' + str(list(df.columns)) + '\n')

    # MAKE MODELS
    (names_models, continuous_features,
     category_features, models, scoring,
     is_continuous, alphas) = make_models(df, df_X, y, y_var_name,
                                          univariates, alphas)

    # MAKE SCATTER MATRIX (after because is_continuous)
    if scatter_matrix:
        if len(unpiped_continuous_features) > 0:
            timeit(plot_scatter_matrix, df, y_continuous=is_continuous, y_var_name=y_var_name, colors=True)
            plt.draw()

    # evaluate each model in turn
    fit_models, results, names, y_hats, errors, seed = [], [], [], [], [], 7

    for name, model in tqdm.tqdm(names_models):
        # if not linear: change df_X to df_X unpiped
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        if name == 'RR' or name == 'LASSO':
            alpha, cv_results = timeit(choose_alpha, df, model,
                                       y_var_name, alphas, kfold, scoring, plot_alphas)
            model = model(alpha)
        else:
            cv_results = timeit(cross_val_score, model, X, y,
                                cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: mean=%f std=%f" % (name, cv_results.mean(),
                                      cv_results.std())
        print(msg)

        # OTHER CROSS VALIDATE METHOD:

        # FIT MODEL WITH ALL DATA
        model.fit(X, y)
        fit_models.append(model)

        # PLOT PREDICTED VS ACTUALS
        if plot_predicted_vs_actuals:
            if is_continuous:
                timeit(plot_predicted_vs_actuals, df,
                    model, y_var_name, sample_limit)
                plt.draw()

        # MAKE BOOTSTRAPS
        if bootstrap_coefs or partial_dep:
            bootstrap_models = bootstrap_train_premade(model, X, y,
                                                       bootstraps=bootstraps,
                                                       fit_intercept=False)

        # PLOT COEFFICIANTS
        if plot_coefs_flag:
            if hasattr(model, "coef_"):
                coefs = model.coef_
                columns = list(df.drop(y_var_name, axis=1).columns)
                while (type(coefs[0]) is list) or (type(coefs[0]) is np.ndarray):
                    coefs = list(coefs[0])
                timeit(plot_coefs, coefs=coefs, columns=columns, graph_name=name)
                plt.draw()

        # PLOT BOOTSTRAP COEFFICIANTS
            if is_continuous:
                if bootstrap_coefs:
                    # PLOT BOOTSTRAP COEFS
                    fig, axs = timeit(plot_bootstrap_coefs, bootstrap_models,
                                      df_X.columns, n_col=4)
                    fig.tight_layout()
                    plt.draw()

        # PLOT FEATURE IMPORTANCES
        if feature_importances:
            if 'feature_importances_' in dir(model):
                timeit(plot_feature_importances, model, df_X)
                plt.draw()

        # PLOT PARTIAL DEPENDENCIES
        if partial_dep:
            timeit(plot_partial_dependences, model, X=df_X_unpiped,
                   var_names=unpiped_continuous_features, y=y,
                   bootstrap_models=bootstrap_models, pipeline=pipeline,
                   n_points=250)
            plt.tight_layout()
            plt.draw()

        # PLOT PREDICTED VS ACTUALS
        if plot_predicteds_vs_actuals:
            plot_continuous_error_graphs(df, y, y_var_name, model,
                                        is_continuous,
                                        sample_limit,
                                        predicteds_vs_actuals=True,
                                        residuals=True)
        df_X = df.drop(y_var_name, axis=1)

        # GET ERROR
        y_hat, error = get_error(name, model, df_X, y, is_continuous)
        y_hats.append(y_hat)
        errors.append(error)

    # --COMPARE MODELS--
    if compare_models:
        choose_box_and_violin_plots(names,
                                    scoring,
                                    compare_models,
                                    results,
                                    is_continuous)
    # ROC CURVE
    if ROC:
        if not is_continuous:
            timeit(plot_rocs, models, df_X, y)
            plt.draw()
    print(f'MAKE SUBSAMPLE TIME: {time() - starttotal}')


    plt.show()
    return names, results, fit_models, pipeline, df_X, y_hats, errors


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
    pass
