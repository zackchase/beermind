import numpy as np

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)