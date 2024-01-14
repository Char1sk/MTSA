import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL


def decomposition(x, s, *args):
    if s == 'moving_average':
        return moving_average(x, *args)
    elif s == 'differential_decomposition':
        return differential_decomposition(x)
    elif s == 'STL_decomposition':
        return STL_decomposition(x, *args)


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


def differential_decomposition(X):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # X: ndarray, (1, time, feature/OT)
    # X: ndarray, (windows_test, seq_len, features)
    X_season = X - np.roll(X, 1, axis=1)
    X_season[0] = 0
    X_trend = X - X_season
    return (X_trend, X_season)


def STL_decomposition(X, seasonal_period):
    """
    A naive implementation of STL
    SUPER SLOW due to non-parallel
    
    Seasonal and Trend decomposition using Loess
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    # X: ndarray, (1, time, feature/OT)
    # X: ndarray, (windows_test, seq_len, features)
    
    # X_trend = np.zeros_like(X)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[2]):
    #         X_trend[i,:,j] = sm.nonparametric.lowess(X[i,:,j], np.arange(X.shape[1]), return_sorted=False)
    # X_detrend = X - X_trend
    
    # X_season = np.zeros_like(X)
    # for p in range(seasonal_period):
    #     interval = np.arange(p, X.shape[1], seasonal_period)
    #     X_season[:,interval,:] = np.mean(X_detrend[:,interval,:], axis=2, keepdims=True)
    
    # X_residual = X_detrend - X_season
    
    X_trend = np.zeros_like(X)
    X_season = np.zeros_like(X)
    X_residual = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            result = STL(X[i,:,j], period=seasonal_period, robust=True).fit()
            X_trend[i,:,j] = np.array(result.trend)
            X_season[i,:,j] = np.array(result.seasonal)
            X_residual[i,:,j] = np.array(result.resid)
    
    return (X_trend, X_season, X_residual)


def X11_decomposition(x, seasonal_period):
    """
    X11 decomposition
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
        residual (numpy.ndarray): Residual component
    """
    
    raise NotImplementedError

