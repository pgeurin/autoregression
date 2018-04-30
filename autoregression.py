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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
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
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
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

#my own growing graphing module
# from galgraphs import simple_spline_specification
# import galgraphs
import os
sys.path.append(os.path.abspath("/Users/macbookpro/Dropbox/Galvanize/cleandata/"))
import cleandata
import importlib.util
spec = importlib.util.spec_from_file_location("galgraphs", "/Users/macbookpro/Dropbox/Galvanize/galgraphs/galgraphs/galgraphs.py")
galgraphs = importlib.util.module_from_spec(spec)
import galgraphs
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

def cross_var_train(df, y_var_name, pipeliner=auto_spline_pipeliner, knots=10):
    df_X = df.drop(y_var_name, axis =1)
    pipeline = pipeliner(df_X,knots)
    train_raw, test_raw = train_test_split(df, test_size=0.33)
    train_X_raw=train_raw.drop(y_var_name, axis =1)
    test_X_raw=test_raw.drop(y_var_name, axis =1)
    pipeline.fit(train_X_raw)
    df_X_train = pipeline.transform(train_X_raw)
    df_X_test = pipeline.transform(test_X_raw)
    X_train, X_test = df_X_train.values, df_X_test.values
    y_train, y_test, y_mean, y_std = galgraphs.standardize_y(train_raw[y_var_name], test_raw[y_var_name])
    return (pipeline, df_X_train, df_X_test, X_train, X_test, y_train, y_test, y_mean, y_std)

def make_ridges(df_X_train, y_train, X_test, alpha_min = 0.000001, alpha_max=1000000):
    ridge_regularization_strengths = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=100)
    ridge_regressions = []
    for alpha in ridge_regularization_strengths:
        ridge = Ridge(alpha=alpha)
        ridge.fit(df_X_train, y_train)
        ridge_regressions.append(ridge)
        y_hat = ridge.predict(X_test)
    return ridge_regressions

def rss(model, X, y):
    y_hat = model.predict(X)
    n = X.shape[0]
    return np.sum((y - y_hat)**2) / n

def train_and_test_error(regressions, X_train, y_train, X_test, y_test):
    alphas = [ridge.alpha for ridge in regressions]
    train_scores = [rss(reg, X_train, y_train) for reg in regressions]
    test_scores = [rss(reg, X_test, y_test) for reg in regressions]
    return pd.DataFrame({
        'train_scores': train_scores,
        'test_scores': test_scores,
    }, index=alphas)

def get_optimal_alpha(train_and_test_errors):
    test_errors = train_and_test_errors["test_scores"]
    optimal_idx = np.argmin(test_errors.values)
    return train_and_test_errors.index[optimal_idx]

def plot_train_and_test_error(ax, train_and_test_errors, alpha=1.0, linewidth=2, legend=True):
    alphas = train_and_test_errors.index
    optimal_alpha = get_optimal_alpha(train_and_test_errors)
    ax.plot(np.log10(alphas), train_and_test_errors.train_scores, label="Train MSE",
            color="blue", linewidth=linewidth, alpha=alpha)
    ax.plot(np.log10(alphas), train_and_test_errors.test_scores, label="Test MSE",
            color="red", linewidth=linewidth, alpha=alpha)
    ax.axvline(x=np.log10(optimal_alpha), color="grey", alpha=alpha)
    ax.set_xlabel(r"$\log_{10}(\alpha)$")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Mean Squared Error vs. Regularization Strength")
    if legend:
        ax.legend()

def make_linear_regression(df, y_var_name, df_test):
    df_X = df.drop(y_var_name, axis = 1)
    df_y = df[y_var_name]
    (continuous_features, category_features) = sort_features(df_X)
    df_X_cont = df_X[continuous_features]
    lr = LinearRegression()
    lr.fit(df_X_cont,df_y)
    df_X_test_cont = df_test[continuous_features]
    y_hat = lr.predict(df_X_test_cont)
    return y_hat

def make_k_folds_ridge(df, y_var_name, pipeliner=auto_spline_pipeliner, knots = 10, num_alphas = 100, n_folds = 10, alpha_min=0.00001, alpha_max = 10000000):
    df_X = df.drop(y_var_name, axis =1)
    trained_pipeline = pipeliner(df_X,knots)
    cv_models = []
    errors = []
    splitter = KFold(n_splits=n_folds)
    ridge_regularization_strengths = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=num_alphas)
    for train_idxs, test_idxs in tqdm.tqdm(splitter.split(df_X), total=n_folds):
        # Split the raw data into train and test
        train_raw_X, test_raw_X,  train_raw_y, test_raw_y  = df_X.iloc[train_idxs], df_X.iloc[test_idxs], df[y_var_name].iloc[train_idxs], df[y_var_name].iloc[test_idxs]
        train_raw, test_raw = df.iloc[train_idxs], df.iloc[test_idxs], 
        # Fit and transform the raw data.

        # All training of the transformers must only touch the training data!
        # %pdb
        trained_pipeline.fit(train_raw_X)
        df_X_train_cv = trained_pipeline.transform(train_raw_X)
        df_X_test_cv = trained_pipeline.transform(test_raw)
        y_train_cv, y_test_cv, y_cv_mean, y_cv_std = galgraphs.standardize_y(train_raw_y, test_raw_y)
        # y_train_cv, y_test_cv = train_raw_y, test_raw_y
        # y_cv_mean = 0
        # y_cv_std = 1

        # Fit all the models at different regularization strengths
        ridge_regressions = []
        for alpha in ridge_regularization_strengths:
            ridge = Ridge(alpha=alpha)
            ridge.fit(df_X_train_cv, y_train_cv)
            ridge_regressions.append(ridge)
        cv_models.append(ridge_regressions)

        # ridge_regressions = make_ridges(df_X_train, y_train, X_test, alpha_min = 0.000001, alpha_max=10000)

        # Calculate the error curves for each CV fold, for each regularization strength
        train_and_test_errors = train_and_test_error(
            ridge_regressions, df_X_train_cv, y_train_cv, df_X_test_cv, y_test_cv)
        errors.append(train_and_test_errors)

        # Calculate the mean errors across all CV folds, for each regularization strength
        train_errors = np.empty(shape=(n_folds, len(ridge_regularization_strengths)))
        for idx, tte in enumerate(errors):
            te = tte['train_scores']
            train_errors[idx, :] = te
        mean_train_errors = np.mean(train_errors, axis=0)

        test_errors = np.empty(shape=(n_folds, len(ridge_regularization_strengths)))
        for idx, tte in enumerate(errors):
            te = tte['test_scores']
            test_errors[idx, :] = te
        mean_test_errors = np.mean(test_errors, axis=0)

        mean_errors = pd.DataFrame({
            'train_scores': mean_train_errors,
            'test_scores': mean_test_errors,
        }, index=ridge_regularization_strengths)
    
    # print (f'test_raw_y = {test_raw_y}')
    # print (f'y_train_cv = {y_train_cv}')
    fig, ax = plt.subplots(figsize=(16, 4))
    # for ttes in errors:
    #     plot_train_and_test_error(ax, ttes, alpha=alpha, legend=False)
    plot_train_and_test_error(ax, mean_errors, linewidth=4, legend=True)
    plt.show()
    alpha = get_optimal_alpha(mean_errors)
    rr_optimized = Ridge(alpha)
    df_X_tranformed = trained_pipeline.transform(df_X)
    # print(df_X_tranformed)
    y_standardized = (df[y_var_name].values.reshape(-1,1) - y_cv_mean) / y_cv_std
    # print (f'df[y_var_name] = {df[y_var_name].values.reshape(-1,1)}')
    # print (f'y_standardized = {y_standardized}')
    rr_optimized.fit(df_X_tranformed.values, y_standardized)
    return (rr_optimized, trained_pipeline, y_cv_mean, y_cv_std)

def fit_predict_model():
    pass

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

def compare_predictions(df, y_var_name, percent_data=None, possible_categories=11, knots=5, univariates=True, bootstraps=50):
    
    if percent_data == None:
        while len(df)>1000:
            print(f"'percent_data' NOT SPECIFIED AND len(df)=({len(df)}) IS > 1000: TAKING A RANDOM %10 OF THE SAMPLE")
            df = df.sample(frac=.1)
    else:
        df = df.sample(frac=percent_data)
    df = cleandata.clean_df(df, y_var_name)
    # REMEMBER OLD DATAFRAME
    df_unpiped = df.copy()
    columns_unpiped = df.columns
    columns_unpiped = list(columns_unpiped)
    columns_unpiped.remove(y_var_name)

    # REMOVE CATEGORICAL VARIABLES THAT HAVE TOO MANY CATEGORIES TO BE USEFUL
    (continuous_features, category_features) = sort_features(df.drop(y_var_name, axis=1))
    for cat in category_features:
        if len(df[cat].unique())>possible_categories:
            df.drop(cat, axis=1)
            print('Too many unique values in categorical feature "' + cat + '", dropping "' + cat + '"')

    # SHOW CORRELATION MATRIX
    plt.matshow(df.corr())
    plt.show()

    # MAKE SCATTER MATRIX
    # KEEP ME: FIX BOOLEAN CASE BEFORE DELETING:
    if len(df) < 300:
        sample_limit = len(df)
    else:
        sample_limit = 300
    if y_var_name in continuous_features:
        continuous_features.remove(y_va r_name)
    while 5 < len(continuous_features):
        plot_sample_df = df[[y_var_name] + continuous_features[:6]].sample(n=sample_limit)
        pd.scatter_matrix(plot_sample_df, figsize=(len(plot_sample_df)*.07,len(plot_sample_df)*.07))
        continuous_features = continuous_features[5:]
    plot_sample_df = df[[y_var_name] + continuous_features].sample(n=sample_limit)
    pd.scatter_matrix(plot_sample_df, figsize=(len(plot_sample_df)*.1,len(plot_sample_df)*.1))
    plt.show()

    print('df columns: ' + str(list(df.columns)))
    # TRANSFORM DATAFRAME
    df_X = df.drop(y_var_name, axis = 1)
    pipeline = auto_spline_pipeliner(df_X, knots=5)
    pipeline.fit(df_X)
    df_X = pipeline.transform(df_X)
    X = df_X.values   
    y = df[y_var_name]
    df = df_X
    df[y_var_name] = y
    print('df columns after transform: ' + str(list(df.columns)))

    # CHOOSE MODELS FOR CONTINUOUS OR CATEGORICAL Y
    names_models = []
    is_continuous = 2 < len(y.unique())
    if is_continuous:
        print ( 'y variable: "' + y_var_name + '" is continuous' )
        if univariates==True:
            galgraphs.plot_many_univariates(df, y_var_name)
            plt.show()
        names_models.append(('LR', LinearRegression())) # LinearRegression as no ._coeff???!
        alpha_range = np.linspace(.0001, 10000,10)
        names_models.append(('RR', RidgeCV(alphas=alpha_range)))
        # names_models.append(('LASSO', LassoCV()))
        # names_models.append(('DT', DecisionTreeRegressor()))
        # names_models.append(('RF', RandomForestRegressor()))
        # names_models.append(('GB', GradientBoostingRegressor()))
        # names_models.append(('SVM', SVC()))
        # evaluate each model in turn
        scoring = 'neg_mean_squared_error'
    else: 
        print ( 'y variable: "' + y_var_name + '" is categorical' )
        names_models.append(('LR', LogisticRegression()))
        # names_models.append(('LDA', LinearDiscriminantAnalysis()))
        # names_models.append(('KNN', KNeighborsClassifier()))
        names_models.append(('DT', DecisionTreeClassifier()))
        # names_models.append(('NB', GaussianNB()))
        # names_models.append(('RF', RandomForestClassifier()))
        # names_models.append(('GB', GradientBoostingClassifier())
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
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        print(model)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: mean=%f std=%f" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        # ADD GRIDSEARCH HERE

        # FIT MODEL WITH ALL DATA
        model.fit(X,y)
        fit_models.append(model)

        # PLOT PREDICTED VS ACTUALS
        plot_sample_df = df.sample(sample_limit)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(name + " Predicteds vs Actuals at " + df.drop(y_var_name, axis = 1).columns[0])
        ax.scatter(df[df.drop(y_var_name, axis = 1).columns[0]], df[y_var_name], color="grey", alpha=0.5)
        ax.scatter(df[df.drop(y_var_name, axis = 1).columns[0]], model.predict(X))
        plt.show()

        # MAKE BOOTSTRAPS
        bootstrap_models = bootstrap_train_premade(model, X, y, bootstraps=bootstraps, fit_intercept=False)

        #PLOT COEFFICIANTS
        if "coef_" in dir(model):
            coefs = model.coef_
            columns=list(df.columns)
            columns.remove(y_var_name)
            while (type(coefs[0]) is list) or (type(coefs[0]) is np.ndarray):
                coefs = list(coefs[0])
            galgraphs.plot_coefs(coefs=coefs, columns=columns, graph_name=name)
            plt.show()

            # PLOT BOOTSTRAP COEFS
            # fig, axs = plot_bootstrap_coefs(bootstrap_models, df_X.columns, n_col=4)
            # fig.tight_layout()
            # plt.show()
        
        # PLOT PARTIAL DEPENDENCIES
        plot_partial_dependences(model, X=df_unpiped.drop(y_var_name, axis=1), var_names=columns_unpiped, y=y, bootstrap_models=bootstrap_models, pipeline=pipeline, n_points=250)
        plt.show()
        # galgraphs.shaped_plot_partial_dependences(model, df, y_var_name)
        # plt.show()

        # PLOT PREDICTED VS ACTUALS
        df_X = df.drop(y_var_name, axis=1)
        y_hat = model.predict(df_X)
        if len(y)>0:
            if len(y) == len(y_hat):
                if is_continuous:
                    (continuous_features, category_features) = sort_features(df_X)
                    galgraphs.plot_many_predicteds_vs_actuals(df_X, continuous_features, y, y_hat.reshape(-1), n_bins=50)
                # galgraphs.plot_many_predicteds_vs_actuals(df_X, category_features, y, y_hat.reshape(-1), n_bins=50)
                # add feature to jitter plot to categorical features
                # add cdf???
                fig, ax = plt.subplots()
                galgraphs.plot_residual_error(ax, df_X.values[:,0].reshape(-1), y.reshape(-1), y_hat.reshape(-1), s=30);
                print(f'{name}: MSE = {np.mean((y_hat-y)**2)}')
            else:
                print ('len(y) != len(y_hat), so no regpressions included' )
        else: 
            print( 'No y, so no regressions included')
    
    # --COMPARE MODELS--
    fig, ax = plt.subplots(2,2, figsize=(20,20))
    ax = ax.flatten()
    fig.suptitle(f'Model Crossval Scores: {scoring}')
    ax[0].set_ylabel(f'{scoring}')

    # BOX PLOTS
    ax[0].boxplot(results, vert=False)
    ax[0].set_yticklabels(names)

    # VIOLIN PLOTS
    ax[1].violinplot(results, vert=False)
    ax[1].set_yticklabels(names)
    
    #BOX PLOTS OF -LOG(ERROR)
    ax[2].boxplot(results, vert=False)
    ax[2].set_yticklabels(names)
    ax[2].set_xlabel(f'{scoring}')
    ax[2].set_xscale('log')
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    #VIOLIN PLOTS OF -LOG(ERROR)
    ax[3].violinplot(results, vert=False)
    ax[3].set_yticklabels(names)
    ax[3].set_xlabel(f'-{scoring}')
    ax[3].set_xscale('log')
    ax[3].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()

    # ROC CURVE
    # print(f'categorical? {str(!is_continuous)}')
    if not is_continuous:
        galgraphs.plot_rocs(models, df_X, y)
        plt.show()
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
