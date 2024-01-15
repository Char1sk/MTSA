# import torch.nn as nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression
from src.utils.decomposition import decomposition


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        # self.model = Model(args)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.model_trend  = LinearRegression(args)
        self.model_season = LinearRegression(args)
        self.decomp_func = args.decomp
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        # NOTE: Ver.1: Decomp -> slide -> fit
        # X_trend, X_season = decomposition(X, self.decomp_func, 24)
        # self.model_trend.fit(X_trend)
        # self.model_season.fit(X_season)
        
        # NOTE: Ver.2: Slide -> decomp -> fit
        # wins = np.concatenate((
        #     [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        # ))
        # wins_trend, wins_season = decomposition(wins, self.decomp_func, 24)
        # self.model_trend.slided_fit(wins_trend)
        # self.model_season.slided_fit(wins_season)
        
        # NOTE: Ver.3: Slide -> SPLIT -> decompXY -> fit12
        wins = np.concatenate((
            [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        ))
        self.slided_fit(wins)
        # X_wins, Y_wins = wins[:, :self.seq_len, :], wins[:, self.seq_len:, :]
        # # Have to get [:2] for (trend, season) and (trend, season, residual)
        # X_wins_trend, X_wins_season = decomposition(X_wins, self.decomp_func, 24)[:2]
        # Y_wins_trend, Y_wins_season = decomposition(Y_wins, self.decomp_func, 24)[:2]
        # self.model_trend.slided_fit(np.concatenate((X_wins_trend, Y_wins_trend), axis=1))
        # self.model_season.slided_fit(np.concatenate((X_wins_season, Y_wins_season), axis=1))
        
        # NOTE: Ver.4: Slide -> SPLIT -> decompX -> fit1 -> fit2
        # wins = np.concatenate((
        #     [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        # ))
        # X_wins, Y_wins = wins[:, :self.seq_len, :], wins[:, self.seq_len:, :]
        # X_wins_trend, X_wins_season = decomposition(X_wins, self.decomp_func, 24)
        # self.model_trend.slided_fit(np.concatenate((X_wins_trend, Y_wins), axis=1))
        # Y_wins_trend_pred = self.model_trend.forecast(X_wins_trend, self.pred_len)
        # self.model_season.slided_fit(np.concatenate((X_wins_season, Y_wins-Y_wins_trend_pred), axis=1))
    
    def slided_fit(self, wins: np.ndarray) -> None:
        # NOTE: Ver.3: Slide -> SPLIT -> decompXY -> fit12
        X_wins, Y_wins = wins[:, :self.seq_len, :], wins[:, self.seq_len:, :]
        # Have to get [:2] for (trend, season) and (trend, season, residual)
        X_wins_trend, X_wins_season = decomposition(X_wins, self.decomp_func, 24)[:2]
        Y_wins_trend, Y_wins_season = decomposition(Y_wins, self.decomp_func, 24)[:2]
        self.model_trend.slided_fit(np.concatenate((X_wins_trend, Y_wins_trend), axis=1))
        self.model_season.slided_fit(np.concatenate((X_wins_season, Y_wins_season), axis=1))
        self.fitted = True
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        X_trend, X_season = decomposition(X, self.decomp_func, 24)[:2]
        Y_trend = self.model_trend.forecast(X_trend, pred_len)
        Y_season = self.model_season.forecast(X_season, pred_len)
        return Y_trend + Y_season
    
    


# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/pdf/2205.13504.pdf
#     """

#     def __init__(self, configs, individual=False):
#         """
#         individual: Bool, whether shared model among different variates.
#         """
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.individual = individual
#         self.channels = configs.enc_in

#         # TODO: implement the following layers

#     def forward(self, x):
#         raise NotImplementedError

#         # TODO: implement the forward pass
