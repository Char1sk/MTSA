import numpy as np


def euclidean(A, B):
    return np.sqrt(np.sum(np.sum((A - B) ** 2, axis=1), axis=1))


def manhattan(A, B):
    return np.linalg.norm(B-A, ord=1, axis=1)


def chebyshev(A, B):
    return np.linalg.norm(B-A, ord=np.inf, axis=1)


# A(1, seq_len) B(windows, seq_len)
# return: (windows,)
def cosine(A, B):
    dots = (A@B.T).squeeze(0)
    normA = np.linalg.norm(A, axis=1)
    normB = np.linalg.norm(B, axis=1)
    return dots / (normA*normB)


def dtw(A, B):
    l = A.shape[1]
    w = B.shape[0]
    ret = np.zeros((w, l+1,l+1))
    ret[:, 0, 1:] = np.inf
    ret[:, 1:, 0] = np.inf
    dist = np.concatenate([np.expand_dims(np.abs(A.T-bl), axis=0) for bl in B], axis=0)
    print(dist.shape)
    for i in range(1, l+1):
        for j in range(1, l+1):
            # dist = np.linalg.norm(A[:,i:i+1]-B[:,j:j+1], axis=1)
            ret[:,i,j] = dist[:,i-1,j-1] + np.min(np.c_[ret[:,i,j-1],ret[:,i-1,j],ret[:,i-1,j-1]], axis=1)
    return ret[:,l,l]


def get_distance(s):
    s_to_f = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'cosine': cosine,
        'dtw': dtw
    }
    return s_to_f[s]
