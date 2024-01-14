import numpy as np

from src.models.base import MLForecastModel

from statsmodels.tsa.forecasting.theta import ThetaModel


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.models = []
    
    def _fit(self, X: np.ndarray) -> None:
        # X: ndarray, (1, time, feature/OT)
        pass
        # if not self.fitted:
        #     for i in range(X.shape[2]):
        #         self.models.append(ThetaModel(X[0,:,i], period=24).fit())
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray, (windows_test, seq_len, features)
        Y = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for j in range(X.shape[2]):
            # model = self.models[j]
            for i in range(X.shape[0]):
                Y[i,:,j] = ThetaModel(X[i,:,j], period=24).fit().forecast(pred_len)
                # Y[i,:,j] = model.forecast(pred_len)
                if i % 100 == 0:
                    print(i, '/', X.shape[0])
        return Y
