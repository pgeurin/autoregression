
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