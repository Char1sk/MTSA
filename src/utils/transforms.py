import numpy as np


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# add other transforms

class NormalizationTransform(Transform):
    def __init__(self, args) -> None:
        pass
    
    def transform(self, data):
        # data: np.ndarray, shape=(n_samples, timesteps, channels) or (windows, channels)
        self.mins = data.min()
        self.maxs = data.max()
        data_t = (data-self.mins) / (self.maxs-self.mins)
        return data_t
    
    def inverse_transform(self, data):
        data_t = data * (self.maxs-self.mins) + self.mins
        return data_t


class StandardizationTransform(Transform):
    def __init__(self, args) -> None:
        pass

    def transform(self, data):
        # data: np.ndarray, shape=(n_samples, timesteps, channels) or (windows, channels)
        self.mean = data.reshape((-1, data.shape[-1])).mean(axis=1)
        self.std = data.reshape((-1, data.shape[-1])).std(axis=1)
        data_t = (data-self.mean) / self.std
        return data_t

    def inverse_transform(self, data):
        data_t = data * self.std + self.mean
        return data_t


class MeanNormalizationTransform(Transform):
    def __init__(self, args) -> None:
        pass

    def transform(self, data):
        self.mean = data.mean()
        self.mins = data.min()
        self.maxs = data.max()
        data_t = (data-self.mean) / (self.maxs-self.mins)
        return data_t

    def inverse_transform(self, data):
        data_t = data * (self.maxs-self.mins) + self.mean
        return data_t


class BoxCoxTransform(Transform):
    def __init__(self, args) -> None:
        # This transform doesnt change SIGNs
        self.lamda = args.boxcox_lambda

    def transform(self, data):
        pos_idx = data>=0
        neg_idx = np.invert(pos_idx)
        data_t = data.copy()
        if self.lamda == 0:
            data_t[pos_idx] = np.log(data_t[pos_idx]+1)
            data_t[neg_idx] = -(np.power(-data_t[neg_idx]+1, 2-self.lamda)-1) / (2-self.lamda)
        elif self.lamda == 2:
            data_t[pos_idx] = (np.power(data_t[pos_idx]+1, self.lamda)-1) / self.lamda
            data_t[neg_idx] = -np.log(-data_t[neg_idx]+1)
        else:
            data_t[pos_idx] = (np.power(data_t[pos_idx]+1, self.lamda)-1) / self.lamda
            data_t[neg_idx] = -(np.power(-data_t[neg_idx]+1, 2-self.lamda)-1) / (2-self.lamda)
        return data_t

    def inverse_transform(self, data):
        pos_idx = data>=0
        neg_idx = np.invert(pos_idx)
        data_t = data.copy()
        if self.lamda == 0:
            data_t[pos_idx] = np.exp(data_t[pos_idx])-1
            data_t[neg_idx] = -np.power((self.lamda-2)*data_t[neg_idx]+1, 1/(2-self.lamda)) + 1
        elif self.lamda == 2:
            data_t[pos_idx] = np.power(self.lamda*data_t[pos_idx]+1, 1/self.lamda) - 1
            data_t[neg_idx] = -np.exp(-data_t[neg_idx])+1
        else:
            data_t[pos_idx] = np.power(self.lamda*data_t[pos_idx]+1, 1/self.lamda) - 1
            data_t[neg_idx] = -np.power((self.lamda-2)*data_t[neg_idx]+1, 1/(2-self.lamda)) + 1
        return data_t


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxcox_lambda', type=float, default=0.0, help='hyper-parameter lambda in BoxCox')
    args = parser.parse_args()
    
    x = (np.random.random((1, 10, 8))-0.5)/0.5
    norms = [
        NormalizationTransform(args),
        StandardizationTransform(args),
        MeanNormalizationTransform(args),
        BoxCoxTransform(args)
    ]
    
    for norm in norms:
        x_t = norm.transform(x)
        x_tt = norm.inverse_transform(x_t)
        print(np.all(np.abs(x-x_tt)<1e-5))
    