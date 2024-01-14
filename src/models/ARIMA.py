import numpy as np

from src.models.base import MLForecastModel

import statsmodels


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        if not self.fitted:
            for i in range(X.shape[2]):
                self.models.append(statsmodels.tsa.arima.model.ARIMA(X[0,:,i]).fit())
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        Y = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for j in range(X.shape[2]):
            model = self.models[j]
            for i in range(X.shape[0]):
                Y[i,:,j] = model.apply(X[i,:,j]).forecast(pred_len)
                if i % 100 == 0:
                    print(i, '/', X.shape[0])
        return Y
