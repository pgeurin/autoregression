import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six.moves
import itertools
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import signal
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_process(size, ar_coefs, ma_coefs, d=0):
    """Simulate a series from an arima model."""
    arma = ArmaProcess(ar_coefs, ma_coefs)
    arma_series = arma.generate_sample(size + d)
    # Integrate d times.
    for i in six.moves.range(d):
        arma_series = np.cumsum(arma_series)
    return pd.Series(arma_series)


def plot_arima_process(ax, size, ar_coefs, ma_coefs, d=0):
    series = arima_process(size, ar_coefs, ma_coefs, d)
    ax.plot(series.index, series)


def plot_series_and_difference(axs, series, title):
    diff = series.diff()
    axs[0].plot(series.index, series)
    axs[0].set_title("Raw Series: {}".format(title))
    axs[1].plot(series.index, diff)
    axs[1].set_title("Series of First Differences: {}".format(title))


def series_and_lagged(series, lag=1):
    truncated = np.copy(series)[lag:]
    lagged = np.copy(series)[:(len(truncated))]
    return truncated, lagged


def compute_autocorrelation(series, lag=1):
    series, lagged = series_and_lagged(series, lag=lag)
    return np.corrcoef(series, lagged)[0, 1]


def plot_series_and_first_differences(timeseries):
    fig, axs = plt.subplots(2, figsize=(14, 4))
    plot_series_and_difference(axs, timeseries, "timeseries")
    plt.tight_layout()

def timestamp_events_to_timeseries(event_timestamp_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id'):
    count_events_per_timestamp_df = event_timestamp_df.groupby([event_col_name, time_col_name]).count()
    timeseries = pd.Series(event_timestamp_df.loc[1][other_col_name].values,
                                event_timestamp_df.loc[1].index)
    timeseries = timeseries.astype('float64')
    timeseries = timeseries.reindex(pd.date_range(min(timeseries.index), max(timeseries.index), freq='D'), fill_value=0)


def plot_series_and_first_differences_over_bound_time(events_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id', num=10, start=None, stop=None):
    if not num:
        if num > 10:
            num = 10
    if not start:
        start = min(events_df[time_col_name])
    if not stop:
        stop = max(events_df[time_col_name])
    events_df[time_col_name] = pd.to_datetime(events_df[time_col_name], errors='coerce')
    count_posts_per_day_df = events_df.groupby([event_col_name, time_col_name]).count()
    min_date, max_date = start, stop
    for i in count_posts_per_day_df.index.get_level_values(0).unique()[:num]:
        timeseries = pd.Series(count_posts_per_day_df.loc[i][other_col_name].values,
                                    count_posts_per_day_df.loc[i].index)
        timeseries = timeseries.astype('float64')
        timeseries = timeseries.reindex(pd.date_range(start, stop, freq='D'), fill_value=0)
        if len(timeseries)>3:
            plot_series_and_first_differences(timeseries)
            plt.show()
    return None

def plot_correlations(timeseries_diff):
    # 1
    plot_lag_correlation(timeseries_diff)
    # 2
    fig, ax = plt.subplots(1, figsize=(14, 3))
    _ = sm.graphics.tsa.plot_acf(timeseries_diff, lags=62, ax=ax)
    # 3
    fig, ax = plt.subplots(1, figsize=(16, 4))
    _ = sm.graphics.tsa.plot_pacf(timeseries_diff, lags=3*52, ax=ax)
    plt.show()
    return None


def plot_lag_correlation(timeseries_diff):
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.flatten()):
        series, lagged = series_and_lagged(timeseries_diff, lag=i)
        autocorr = compute_autocorrelation(timeseries_diff, lag=i)
        ax.scatter(series, lagged, alpha=0.5)
        ax.set_title("Lag {0} AC: {1:2.2f}".format(i, autocorr))
    plt.tight_layout()
    plt.show()
    return None


def auto_regressive_process(size, coefs, init=None):
    """Generate an autoregressive process with Gaussian white noise.  The
    implementation is taken from here:

      http://numpy-discussion.10968.n7.nabble.com/simulate-AR-td8236.html

    Exaclty how lfilter works here takes some pen and paper effort.
    """
    coefs = np.asarray(coefs)
    if init is None:
        init = np.array([0]*len(coefs))
    else:
        init = np.asarray(init)
    init = np.append(init, np.random.normal(size=(size - len(init))))
    assert(len(init) == size)
    a = np.append(np.array([1]), -coefs)
    b = np.array([1])
    return pd.Series(signal.lfilter(b, a, init))


def format_list_of_floats(L):
    return ["{0:2.2f}".format(f) for f in L]


def sim_data(model, timeseries_diff):
    fig, ax = plt.subplots(4, figsize=(14, 8))
    ax[0].plot(timeseries_diff.index, timeseries_diff)
    ax[0].set_title("First Differences of Series Data")
    for i in range(1, 4):
        simulated_data = auto_regressive_process(
                                    len(timeseries_diff),
                                    list(model.params)[1:])
        simulated_data.index = timeseries_diff.index
        ax[i].plot(simulated_data.index, simulated_data)
        ax[i].set_title("Simulated Data Fit")
    plt.tight_layout()
    plt.show()
    return None

def plot_forecast_interval(model, timeseries):
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax.plot(timeseries.index, timeseries)
    fig = model.plot_predict('2017-12-11', '2020',
                                      dynamic=True, ax=ax, plot_insample=False)
    _ = ax.legend().get_texts()[0].set_text("95% Prediction Interval")
    _ = ax.legend(loc="lower left")
    _ = ax.set_title("Series Forcasts from ARIMA Model")
    plt.show()
    return None


def make_model_residuals(model):
    fig, ax = plt.subplots(1, figsize=(14, 3))
    ax.plot(model.resid.index, model.resid)
    ax.set_title("Residuals from Model")
    plt.tight_layout()
    plt.show()
    return None


def try_arimas(timeseries):
    parameters = [
        {'AR': 3, 'MA': 0},
        {'AR': 2, 'MA': 0},
        {'AR': 4, 'MA': 0},
        {'AR': 3, 'MA': 1}
    ]
    models = {}
    for params in parameters:
        models[(params['AR'], params['MA'])] = ARIMA(timeseries, order=(params['AR'], 1, params['MA'])).fit()
    for model_params in models:
        print("ARIMA({}, 1, {}) AIC: {}".format(model_params[0], model_params[1], models[model_params].aic))
    return None

def plot_prediction(timeseries, seasonal_model):
    fig, ax = plt.subplots(1, figsize=(16, 4))
    ax.plot(timeseries.index, timeseries)
    preds = seasonal_model.predict('2018-03-17', '2018-04-17',
                       dynamic=True, ax=ax, plot_insample=False)
    ax.plot(preds.index, preds)
    ax.set_title("Forecasts")
    plt.show()
    return None


def make_arema_prediction(timeseries):
    timeseries = fill_and_float_timeseries(timeseries, freq='D')
    timeseries_diff = timeseries.diff()[1:]
    plot_correlations(timeseries_diff)
    model = ARIMA(timeseries, order=(3, 1, 0)).fit()
    print("ARIMA(3, 1, 0) coefficients :\n  Intercept {0:2.2f}\n  AR {1}".format(
        model.params[0],
            format_list_of_floats(list(model.params[1:]))
        ))
    plot_forecast_interval(model, timeseries)
    make_model_residuals(model)
    fig, ax = plt.subplots(1, figsize=(14, 3))
    _ = sm.graphics.tsa.plot_acf(model.resid, lags=50, ax=ax)
    try_arimas(timeseries)
    seasonal_model = SARIMAX(timeseries, order=(1, 1, 0), seasonal_order=(1, 0, 0, 52)).fit()
    print("ARIMA(3, 1, 0) coefficients :\n  Intercept {0:2.2f}\n  AR {1}".format(
    seasonal_model.params[0],
        format_list_of_floats(list(model.params[1:]))
    ))
    plot_prediction(timeseries, seasonal_model)
    sim_data(model, timeseries_diff)
    return model, seasonal_model


def fill_and_float_timeseries(timeseries, freq='D'):
    timeseries = timeseries.astype('float64')
    return timeseries.reindex(pd.date_range(min(timeseries.index), max(timeseries.index), freq=freq), fill_value=0)


# def main():
#     posts_df = pd.read_csv('../data/posts.csv')
#     posts_df['date'] = pd.to_datetime(posts_df['date'], errors='coerce')
#     plot_series_and_first_differences_over_bound_time(posts_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id', num=10, start=pd.to_datetime("2012"), stop=pd.to_datetime("2018"))
#     count_posts_per_day_df = posts_df.groupby(['classroom_id', 'date']).count()
#     class_1_count_posts_per_day = count_posts_per_day_df.loc[1]
#     timeseries = pd.Series(class_1_count_posts_per_day['id'].values,
#                            class_1_count_posts_per_day.index)
#     timeseries = fill_and_float_timeseries(timeseries, freq='D')
#     # timeseries = timestamp_events_to_timeseries(event_timestamp_df, event_col_name='classroom_id', time_col_name='date', other_col_na
#     make_arema_prediction(timeseries)
#
#
# if __name__ == "__main__":
#     main()


def main():
    posts_df = pd.read_csv('../data/posts.csv')
    posts_df['date'] = pd.to_datetime(posts_df['date'], errors='coerce')
    # plot_series_and_first_differences_over_bound_time(posts_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id', num=10, start=pd.to_datetime("2014"), stop=pd.to_datetime("2018"))
    # plot_series_and_first_differences_over_bound_time(posts_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id', num=10)
    plot_series_and_first_differences_over_bound_time(posts_df, event_col_name='classroom_id', time_col_name='date', other_col_name='id', num=10, start=pd.to_datetime("2012"), stop=pd.to_datetime("2018"))
    count_posts_per_day_df = posts_df.groupby(['classroom_id', 'date']).count()
    first_few_classrooms = list(count_posts_per_day_df.index.get_level_values(0).unique())[:3]
    # for classroom_id in first_few_classrooms:
    #     print(classroom_id)
    #     class_n_count_posts_per_day = count_posts_per_day_df.loc[classroom_id]
    #     timeseries = pd.Series(class_n_count_posts_per_day['id'].values,
    #                            class_n_count_posts_per_day.index)
    #     timeseries = fill_and_float_timeseries(timeseries, freq='D')
    #     # timeseries = timestamp_events_to_timeseries(event_timestamp_df, event_col_name='classroom_id', time_col_name='date', other_col_na
    #     make_arema_prediction(timeseries)
    class_1_count_posts_per_day = count_posts_per_day_df.loc[1]
    timeseries = pd.Series(class_1_count_posts_per_day['id'].values,
                           class_1_count_posts_per_day.index)
    timeseries = fill_and_float_timeseries(timeseries, freq='D')
    # timeseries = timestamp_events_to_timeseries(event_timestamp_df, event_col_name='classroom_id', time_col_name='date', other_col_na
    make_arema_prediction(timeseries)


if __name__ == "__main__":
    main()
