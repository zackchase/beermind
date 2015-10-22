import numpy as np
import theano

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
