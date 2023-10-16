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

# TODO: add other transforms

class NormalizationTransform(Transform):
    def __init__(self, args) -> None:
        pass
    
    def transform(self, data):
        # data: np.ndarray, shape=(n_samples, timesteps, channels)
        self.mins = data.min(axis=1)
        self.maxs = data.max(axis=1)
        data_t = (data-self.mins) / (self.maxs-self.mins)
        return data_t
    
    def inverse_transform(self, data):
        data_t = data * (self.maxs-self.mins) + self.mins
        return data_t


class StandardizationTransform(Transform):
    def __init__(self, args) -> None:
        pass

    def transform(self, data):
        # data: np.ndarray, shape=(n_samples, timesteps, channels)
        self.mean = data.mean(axis=1)
        self.std = data.std(axis=1)
        data_t = (data-self.mean) / self.std
        return data_t

    def inverse_transform(self, data):
        data_t = data * self.std + self.mean
        return data_t


class MeanNormalizationTransform(Transform):
    def __init__(self, args) -> None:
        pass

    def transform(self, data):
        self.mean = data.mean(axis=1)
        self.mins = data.min(axis=1)
        self.maxs = data.max(axis=1)
        data_t = (data-self.mean) / (self.maxs-self.mins)
        return data_t

    def inverse_transform(self, data):
        data_t = data * (self.maxs-self.mins) + self.mean
        return data_t

# TODO: -x
class BoxCoxTransform(Transform):
    def __init__(self, args) -> None:
        self.lamda = args.boxcox_lambda

    def transform(self, data):
        self.pos_idx = data>=0
        self.neg_idx = np.invert(self.pos_idx)
        data_t = data.copy()
        if self.lamda == 0:
            data_t[self.pos_idx] = np.log(data_t[self.pos_idx]+1)
            data_t[self.neg_idx] = -(np.power(-data_t[self.neg_idx]+1, 2-self.lamda)-1) / (2-self.lamda)
        elif self.lamda == 2:
            data_t[self.pos_idx] = (np.power(data_t[self.pos_idx]+1, self.lamda)-1) / self.lamda
            data_t[self.neg_idx] = -np.log(-data_t[self.neg_idx]+1)
        else:
            data_t[self.pos_idx] = (np.power(data_t[self.pos_idx]+1, self.lamda)-1) / self.lamda
            data_t[self.neg_idx] = -(np.power(-data_t[self.neg_idx]+1, 2-self.lamda)-1) / (2-self.lamda)
        return data_t

    def inverse_transform(self, data):
        data_t = data.copy()
        if self.lamda == 0:
            data_t[self.pos_idx] = np.exp(data_t[self.pos_idx])-1
            data_t[self.neg_idx] = -np.power((self.lamda-2)*data_t[self.neg_idx]+1, 1/(2-self.lamda)) + 1
        elif self.lamda == 2:
            data_t[self.pos_idx] = np.power(self.lamda*data_t[self.pos_idx]+1, 1/self.lamda) - 1
            data_t[self.neg_idx] = -np.exp(-data_t[self.neg_idx])+1
        else:
            data_t[self.pos_idx] = np.power(self.lamda*data_t[self.pos_idx]+1, 1/self.lamda) - 1
            data_t[self.neg_idx] = -np.power((self.lamda-2)*data_t[self.neg_idx]+1, 1/(2-self.lamda)) + 1
        return data_t
