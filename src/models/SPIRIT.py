import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

from src.models.base import MLForecastModel
from src.models.DLinear import DLinear


class SPIRIT(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_components = args.n_components
        self.pca = PCA(self.n_components)
        self.model = DLinear(args) # But with args.individual=True
    
    def _fit(self, X: np.ndarray) -> None:
        Xlow = np.expand_dims(self.pca.fit_transform(X.squeeze(0)), 0)
        self.model.fit(Xlow)
    
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        Xlow = np.zeros((X.shape[0], X.shape[1], self.n_components))
        for i in range(X.shape[0]):
            Xlow[i,:,:] = self.pca.transform(X[i,:,:])
        Ylow = self.model.forecast(Xlow, pred_len)
        Y = self.pca.inverse_transform(Ylow)
        return Y
