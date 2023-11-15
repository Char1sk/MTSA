import numpy as np

def moving_average(X, seasonal_period):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # X: ndarray, (1, time, feature/OT)
    # X: ndarray, (windows_test, seq_len, features)
    PERIOD = seasonal_period
    LSHIFT = PERIOD//2
    RSHIFT = PERIOD-LSHIFT
    # X_front = X[:,:1,:].repeat(LSHIFT, axis=1)
    # X_back = X[:,-1:,:].repeat(RSHIFT, axis=1)
    # X_pad = np.concatenate((X_front, X, X_back), axis=1)
    X_pad = np.pad(X, ((0,0),(LSHIFT,RSHIFT),(0,0)), mode='edge')
    times = np.arange(X.shape[1]).reshape(-1, 1)
    interval = np.arange(PERIOD).reshape(1, -1)
    indexes = times + interval
    X_trend = X_pad[:,indexes,:].mean(axis=2)
    X_season = X - X_trend
    return (X_trend, X_season)


def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """

    raise NotImplementedError

