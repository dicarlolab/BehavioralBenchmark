from BehavioralBenchmark import extrapolation

__author__ = 'ardila'

import numpy as np


def test_basic_exponential_fit():
    X = np.arange(100)
    Y = np.exp(X)


    model, error  = extrapolation.best_model(X[::2],Y[::2],X[1::2], Y[1::2])

    assert model.__class__.__name__ == 'PositiveExponentialModel'

    X = np.arange(100)
    Y = np.exp(-X)


    model, error  = extrapolation.best_model(X[::2],Y[::2],X[1::2], Y[1::2])

    assert model.__class__.__name__ == 'NegativeExponentialModel'

test_basic_exponential_fit()