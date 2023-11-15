# import torch.nn as nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.models.baselines import LinearRegression
from src.utils.decomposition import moving_average


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        # self.model = Model(args)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.model_trend  = LinearRegression(args)
        self.model_season = LinearRegression(args)
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        X_trend, X_season = moving_average(X, 24)
        self.model_trend.fit(X_trend)
        self.model_season.fit(X_season)
        # wins = np.concatenate((
        #     [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        # ))
        # wins_trend, wins_season = moving_average(wins, 24)
        # self.model_trend.slided_fit(wins_trend)
        # self.model_season.slided_fit(wins_season)
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        X_trend, X_season = moving_average(X, 24)
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
