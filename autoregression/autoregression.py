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
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn import model_selection
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, RidgeClassifierCV
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
# from autoregression import cleandata
# from autoregression import galgraphs
from galgraphs import (sort_features,
                       simple_spline_specification,
                       plot_many_univariates,
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


def choose_box_and_violin_plots(names, scoring, compare_models, results):
    if is_continuous:
        negresults = []
        for i, result in enumerate(results):
            negresults.append(-1*result)
        timeit(plot_box_and_violins, names, scoring, negresults)
    else:
        timeit(plot_box_and_violins, names, scoring, results)
    plt.show()
    return None


def auto_spline_pipeliner(df_X, knots=10):
    (continuous_features, category_features) = sort_features(df_X)
    # print(continuous_features)
    # print(category_features)
    continuous_pipelet = []
    category_pipelet = []
    for name in continuous_features:
        knotspace = list(np.linspace(df_X[name].min(), df_X[name].max(), knots))
        continuous_pipelet.append((name+'_fit', simple_spline_specification(name, knotspace)))
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
        return var == level
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


def timeit(func, *args, **kwargs):
    print(f'Running: {func.__name__.upper()} ...')
    start = time()
    answers = func(*args, **kwargs)
    print(f'{func.__name__.upper()} TIME: {time() - start}')
    return answers


def make_cont_models(alphas):
    names_models = []
    # names_models.append(('LR', LinearRegression()))
    names_models.append(('RR', RidgeCV(alphas=alphas)))
    names_models.append(('LASSO', LassoCV(alphas=alphas)))
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
    names_models.append(('RR', RidgeClassifierCV(alphas=alphas)))
    names_models.append(('KNN', KNeighborsClassifier()))
    names_models.append(('DT', DecisionTreeClassifier()))
    # names_models.append(('NB', GaussianNB()))
    names_models.append(('RF', RandomForestClassifier()))
    names_models.append(('GB', GradientBoostingClassifier()))
    # names_models.append(('DT', AdaBoostClassifier()))
    # names_models.append(('SVM', SVC()))
    return names_models

def make_models(df, df_X, y, y_var_name, univariates):
    """CHOOSE MODELS FOR CONTINUOUS OR CATEGORICAL Y, make the Models"""
    alphas = np.logspace(start=-2, stop=5, num=5)
    print(len(y.unique()))
    (continuous_features, category_features) = sort_features(df_X)
    is_continuous = (y_var_name in continuous_features)
    if is_continuous:
        print('Y VARIABLE: "' + y_var_name + '" IS CONTINUOUS')
        print()
        if univariates:
            plot_many_univariates(df, y_var_name)
            plt.show()
        names_models = make_cont_models(alphas)
        scoring = 'neg_mean_squared_error'
    else:
        print ( 'Y VARIABLE: "' + y_var_name + '" IS CATEGORICAL' )
        print()
        names_models = make_cat_models(alphas)
        scoring = 'accuracy'
    models = [x[1] for x in names_models]
    return names_models, continuous_features, category_features, models, scoring, is_continuous

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
    df_X = df.drop(y_var_name, axis = 1)
    pipeline = auto_spline_pipeliner(df_X, knots=5)
    pipeline.fit(df_X)
    df_X = pipeline.transform(df_X)
    X = df_X.values
    y = df[y_var_name]
    df = df_X
    df[y_var_name] = y
    return df, df_X, X, y, pipeline


def plot_continuous_error_graphs(df, y_var_name, model, is_continuous, predicteds_vs_actuals=True, residuals=True):
    df_X_sample = df.sample(sample_limit).drop(y_var_name, axis=1)
    y_hat_sample = model.predict(df_X_sample)
    if is_continuous:
        if len(y)>0:
            if len(y) == len(y_hat_sample):
                if predicteds_vs_actuals:
                    (continuous_features, category_features) = sort_features(df_X_sample)
                    timeit(plot_many_predicteds_vs_actuals, df_X_sample, continuous_features, y, y_hat_sample.reshape(-1), n_bins=50)
                    plt.show()
                    # plot_many_predicteds_vs_actuals(df_X_sample, category_features, y, y_hat_sample.reshape(-1), n_bins=50)
                    # add feature to jitter plot to categorical features
                    # add cdf???
                if residuals:
                    fig, ax = plt.subplots()
                    timeit(plot_residual_error, ax, df_X_sample.values[:,0].reshape(-1), y.reshape(-1), y_hat_sample.reshape(-1), s=30)
                    plt.show()
            else:
                print('len(y) != len(y_hat), so no regressions included' )
        else:
            print('No y, so no regressions included')
    return None


def get_error(model, df_X, y, is_continuous):
    if is_continuous:
        y_hat = model.predict(df_X)
        print(f'{name}: MSE = {np.mean((y_hat-y)**2)}')
    else:
        if 'predict_proba' in dir(model):
            y_hat = model.predict_proba(df_X)[:,0]
            logloss = np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            print(f'{name}: logloss = {logloss}')
        if 'decision_function' in dir(model):
            d = model.decision_function(df_X)[0]
            y_hat = np.exp(d) / np.sum(np.exp(d))
            print(f'{name}: logloss = {np.mean((y_hat-y)**2)}')
        return y_hat


def compare_predictions(df, y_var_name, percent_data=None,
                        category_limit=11, knots=3, corr_matrix=True,
                        scatter_matrix=True, bootstrap_coefs=True,
                        feature_importances=True,
                        partial_dep=True, actual_vs_predicted=True,
                        residuals=True, univariates=True, compare_models=True,
                        ROC=True, bootstraps=10):
    """Takes dataframe
        INPUT:
            name:
                string, a feature name to spline
            knots:
                int, number knots (divisions) which are divisions between splines.
        OUTPUT:
            pipeline
    """
    starttotal = time()
    df = cleandata.rename_columns(df)
    y_var_name = stringcase.snakecase(y_var_name).replace('__', '_').replace('__', '_')
    df = timeit(take_subsample, df, percent_data)
    df = timeit(cleandata.clean_df, df, y_var_name)
    sample_limit = make_sample_limit(df)

    # REMEMBER OLD DATAFRAME

    df_unpiped, df_X_unpiped = df.copy(), df.copy().drop(y_var_name, axis=1)
    (unpiped_continuous_features, unpiped_category_features) = sort_features(df_X_unpiped)
    columns_unpiped = df_X_unpiped.columns

    # REMOVE CATEGORICAL VARIABLES THAT HAVE TOO MANY CATEGORIES TO BE USEFUL
    df = cleandata.drop_category_exeeding_limit(df, y_var_name, category_limit)

    # SHOW CORRELATION MATRIX
    if corr_matrix:
        timeit(plt.matshow, df.sample(sample_limit).corr())

    # MAKE SCATTER MATRIX
    if scatter_matrix:
        timeit(plot_scatter_matrix, df, y_var_name)
        plt.show()

    # TRANSFORM DATAFRAME
    print('DF COLUMNS: \n' + str(list(df.columns)) + '\n')
    df, df_X, X, y, pipeline = use_spline(df, y_var_name)
    print('DF COLUMNS AFTER TRANSFORM: \n' + str(list(df.columns)) + '\n')

    #MAKE MODELS
    (names_models, continuous_features, category_features, models, scoring, is_continuous) = make_models(df, df_X, y, y_var_name, univariates)

    # evaluate each model in turn
    fit_models, results, names, seed = [], [], [], 7

    for name, model in tqdm.tqdm(names_models):
        #if not linear: change df_X to df_X unpiped
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = timeit(cross_val_score, model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: mean=%f std=%f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        plt.show()

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
        # plot_solution_paths(ax, ridge_regressions)


        # ADD GRIDSEARCH HERE

        # FIT MODEL WITH ALL DATA
        model.fit(X,y)
        fit_models.append(model)

        # PLOT PREDICTED VS ACTUALS
        if is_continuous:
            timeit(plot_predicted_vs_actuals, df, model, y_var_name, sample_limit)
            plt.show()

        # MAKE BOOTSTRAPS
        if bootstrap_coefs or partial_dep:
            bootstrap_models = bootstrap_train_premade(model, X, y, bootstraps=bootstraps, fit_intercept=False)

        # PLOT COEFFICIANTS

        if hasattr(model, "coef_"):
            coefs = model.coef_
            columns = list(df.columns.drop('y_var_name', axis=1))
            while (type(coefs[0]) is list) or (type(coefs[0]) is np.ndarray):
                coefs = list(coefs[0])
            timeit(plot_coefs, coefs=coefs, columns=columns, graph_name=name)
            plt.show()

        # PLOT BOOTSTRAP COEFFICIANTS
            if is_continuous:
                if bootstrap_coefs:
                    # PLOT BOOTSTRAP COEFS
                    fig, axs = timeit(plot_bootstrap_coefs, bootstrap_models, df_X.columns, n_col=4)
                    fig.tight_layout()
                    plt.show()

        # PLOT FEATURE IMPORTANCES
        if feature_importances:
            if 'feature_importances_' in dir(model):
                timeit(plot_feature_importances, model, df_X)
                plt.show()

        # PLOT PARTIAL DEPENDENCIES
        if partial_dep:
            timeit(plot_partial_dependences, model, X=df_X_unpiped, var_names=unpiped_continuous_features, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            plt.tight_layout()
            # plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            # galgraphs.plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
            plt.show()
            # hot_categorical_vars = [column for column in df.columns if (len(df[column].unique()) == 2)]
            # galgraphs.shaped_plot_partial_dependences(model, df[[y_var_name]+hot_categorical_vars], y_var_name)
            plt.show()

        # PLOT PREDICTED VS ACTUALS
        # Sample no matter what
        plot_continuous_error_graphs(df, y_var_name, model, is_continuous,
                                     predicteds_vs_actuals=True,
                                     residuals=True)

        df_X = df.drop(y_var_name, axis=1)

        # GET ERROR
        y_hat = get_error(model, df_X, y, is_continuous)

    # --COMPARE MODELS--
    if compare_models:
        choose_box_and_violin_plots(names, scoring, compare_models, results)

    # ROC CURVE
    if ROC:
        if not is_continuous:
            timeit(plot_rocs, models, df_X, y)
            plt.show()

    print(f'MAKE SUBSAMPLE TIME: {time() - starttotal}')

    return names, results, models, pipeline, df_X

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
