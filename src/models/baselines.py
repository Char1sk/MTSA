import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.utils.decomposition import moving_average

# Import Line6 instead of Line8 when Testing
if __name__ == '__main__':
    from base import MLForecastModel
else:
    from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len, X.shape[2]))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=1).reshape(X.shape[0], 1, X.shape[2])
        return np.repeat(mean, pred_len, axis=1)

# TODO: add other models based on MLForecastModel

class LinearRegression(MLForecastModel):
    def __init__(self, args) -> None:
        # x: (1, seq_len)
        # y: (1, pred_len)
        # w: (seq_len, pred_len)
        # Predict a Linear layer, seq->pred
        # Instead of feature->OT
        # Cause _forecast gives NO FEATURES
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.individual = args.individual
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        wins = np.concatenate((
            # [sliding_window_view(v, self.seq_len+self.pred_len) for v in X[:,:,-1]]
            [sliding_window_view(v, (self.seq_len+self.pred_len, X.shape[2])).squeeze(1) for v in X[:,:,:]]
        ))
        self.slided_fit(wins)
        # if self.individual:
        #     # x: ndarray, (windows_train, seq_len, features)
        #     x = wins[:, :self.seq_len, :]
        #     y = wins[:, self.seq_len:, :]
        #     self.ws = np.zeros((X.shape[2], self.seq_len, self.pred_len))
        #     for i in range(X.shape[2]):
        #         xi, yi = x[:, :, i], y[:, :, i]
        #         self.ws[i, :, :] = np.linalg.inv(xi.T @ xi) @ xi.T @ yi
        # else:
        #     # x: ndarray, (windows_train * features, seq_len)
        #     # to minimize compound loss of all features, view features as samples
        #     x = wins[:, :self.seq_len, :].transpose(0,2,1).reshape(-1, self.seq_len)
        #     y = wins[:, self.seq_len:, :].transpose(0,2,1).reshape(-1, self.pred_len)
        #     self.w = np.linalg.inv(x.T @ x) @ x.T @ y
    
    def slided_fit(self, wins: np.ndarray) -> None:
        # X: ndarray, (windows_test, seq_len, features)
        if self.individual:
            # x: ndarray, (windows_train, seq_len, features)
            x = wins[:, :self.seq_len, :]
            y = wins[:, self.seq_len:, :]
            self.ws = np.zeros((wins.shape[2], self.seq_len, self.pred_len))
            for i in range(wins.shape[2]):
                xi, yi = x[:, :, i], y[:, :, i]
                self.ws[i, :, :] = np.linalg.pinv(xi.T @ xi) @ xi.T @ yi
        else:
            # x: ndarray, (windows_train * features, seq_len)
            # to minimize compound loss of all features, view features as samples
            x = wins[:, :self.seq_len, :].transpose(0,2,1).reshape(-1, self.seq_len)
            y = wins[:, self.seq_len:, :].transpose(0,2,1).reshape(-1, self.pred_len)
            self.w = np.linalg.pinv(x.T @ x) @ x.T @ y
        self.fitted = True
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        if self.individual:
            Y = np.zeros((X.shape[0], pred_len, X.shape[2]))
            for i in range(X.shape[2]):
                Y[:, :, i] = X[:, :, i] @ self.ws[i, :, :]
        else:
            n_win, seq_len, n_feature = X.shape
            X = X.transpose(0,2,1).reshape(-1, seq_len)
            Y = (X @ self.w).reshape(n_win, n_feature, pred_len).transpose(0, 2, 1)
        return Y
    


class ExponantialSmoothing(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.lamda = args.es_lambda
    
    def _fit(self, X: np.ndarray) -> None:
        pass
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, OT*seq_len)
        pred = X[:,:1]
        for c in range(1, X.shape[1]+1):
            pred = (1-self.lamda)*X[:,c-1:c] + self.lamda*pred
        return np.tile(pred, (1, pred_len))


if __name__ == '__main__':
    import argparse
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--es_lambda', type=float, default=0.2, help='hyper-parameter lambda in ES')
    args = parser.parse_args()
    # es test
    es = ExponantialSmoothing(args)
    X = np.random.random((10, 5))
    y = es._forecast(X, 3)
    print(X)
    print(y)
    print(X.shape, y.shape)
