import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.TsfKNN import TsfKNN
from src.models.DLinear import DLinear
from src.models.ARIMA import ARIMA
from src.models.ThetaMethod import ThetaMethod
from src.models.baselines import ZeroForecast, MeanForecast, LinearRegression, ExponantialSmoothing
from src.utils.decomposition import decomposition


def get_model(args, name):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LinearRegression': LinearRegression,
        'ExponantielSmoothing': ExponantialSmoothing,
        'TsfKNN': TsfKNN,
        'DLinear': DLinear,
        'ARIMA': ARIMA,
        'ThetaMethod': ThetaMethod,
        'ResidualModel': ResidualModel
    }
    return model_dict[name](args)


class ResidualModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        # self.model_trend  = TsfKNN(args)
        # self.model_season = LinearRegression(args)
        # self.model_residual = MeanForecast(args)
        self.model_trend    = get_model(args, args.model_t)
        self.model_season   = get_model(args, args.model_s)
        self.model_residual = get_model(args, args.model_r)
        self.decomp_func = args.decomp
        self.residual = args.residual
        self.residual_mode = args.residual_mode
        print(self.residual_mode)
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        wins = np.concatenate((
            [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        ))
        X_wins, Y_wins = wins[:, :self.seq_len, :], wins[:, self.seq_len:, :]
        # Have to get [:2] for (trend, season) and (trend, season, residual)
        # X_wins_trend, X_wins_season = decomposition(X_wins, self.decomp_func, 24)[:2]
        # Y_wins_trend, Y_wins_season = decomposition(Y_wins, self.decomp_func, 24)[:2]
        X_wins_trend, X_wins_season, X_wins_residual = decomposition(X_wins, self.decomp_func, 24)
        if not self.residual:
            Y_wins_trend, Y_wins_season, Y_wins_residual = decomposition(Y_wins, self.decomp_func, 24)
            # self.model_trend._set_Xs(np.concatenate((X_wins_trend, Y_wins_trend), axis=1))
            self.model_trend.fit(np.concatenate((X_wins_trend, Y_wins_trend), axis=1))
            # self.model_trend.slided_fit(np.concatenate((X_wins_trend, Y_wins_trend), axis=1))
            self.model_season.fit(np.concatenate((X_wins_season, Y_wins_season), axis=1))
            # self.model_season.slided_fit(np.concatenate((X_wins_season, Y_wins_season), axis=1))
            self.model_residual.fit(np.concatenate((X_wins_residual, Y_wins_residual), axis=1))
            # self.model_residual.slided_fit(np.concatenate((X_wins_residual, Y_wins_residual), axis=1))
        else:
            if self.residual_mode == 'trend_first':
                self.model_trend.fit(np.concatenate((X_wins_trend, Y_wins), axis=1))
                Y_wins_trend_pred = self.model_trend.forecast(X_wins_trend, self.pred_len)
                self.model_season.fit(np.concatenate((X_wins_season, Y_wins-Y_wins_trend_pred), axis=1))
                Y_wins_season_pred = self.model_season.forecast(X_wins_season, self.pred_len)
                self.model_residual.fit(np.concatenate((X_wins_residual, Y_wins-Y_wins_trend_pred-Y_wins_season_pred), axis=1))
            elif self.residual_mode == 'season_first':
                self.model_season.fit(np.concatenate((X_wins_season, Y_wins), axis=1))
                Y_wins_season_pred = self.model_season.forecast(X_wins_season, self.pred_len)
                self.model_trend.fit(np.concatenate((X_wins_trend, Y_wins-Y_wins_season_pred), axis=1))
                Y_wins_trend_pred = self.model_trend.forecast(X_wins_trend, self.pred_len)
                self.model_residual.fit(np.concatenate((X_wins_residual, Y_wins-Y_wins_trend_pred-Y_wins_season_pred), axis=1))
        
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        # X_trend, X_season = decomposition(X, self.decomp_func, 24)[:2]
        X_trend, X_season, X_residual = decomposition(X, self.decomp_func, 24)
        Y_trend = self.model_trend.forecast(X_trend, pred_len)
        Y_season = self.model_season.forecast(X_season, pred_len)
        Y_residual = self.model_residual.forecast(X_residual, pred_len)
        # return Y_trend + Y_season
        return Y_trend + Y_season + Y_residual
