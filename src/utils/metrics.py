import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    return np.mean(np.abs(target-predict))


def mape(predict, target):
    return 100*np.mean(np.abs(target-predict)/np.abs(target))


def smape(predict, target):
    return 200*np.mean(np.abs(target-predict)/(np.abs(target)+np.abs(predict)))


def mase(predict, target, m):
    return np.mean( np.abs(target-predict) / np.mean(np.abs(target[m:]-target[:-m])) )
