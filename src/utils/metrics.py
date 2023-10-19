import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# implement the metrics
def mae(predict, target):
    return np.mean(np.abs(target-predict))


def mape(predict, target):
    # # Avoid target==0 & divide by zero
    # print(target.shape, np.argwhere(target==0).shape)
    # print(target.nonzero())
    predict_nonzero = predict[target!=0]
    target_nonzero = target[target!=0]
    return 100*np.mean(np.abs(target_nonzero-predict_nonzero)/np.abs(target_nonzero))


def smape(predict, target):
    # print(np.any(np.isnan(np.abs(target-predict))), np.any(np.isnan(np.abs(target)+np.abs(predict))), np.any(np.isnan(np.abs(target-predict)/(np.abs(target)+np.abs(predict)))))
    predict_nonzero = predict.copy()
    predict_nonzero[predict==0] += 1e-6
    target_nonzero = target.copy()
    target_nonzero[target==0] += 1e-6
    return 200*np.mean(np.abs(target_nonzero-predict_nonzero)/(np.abs(target_nonzero)+np.abs(predict_nonzero)))


def mase(predict, target):
    m = 24
    return np.mean( np.abs(target-predict) / np.mean(np.abs(target[m:]-target[:-m])) )
