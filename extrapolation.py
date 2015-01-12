__author__ = 'ardila'
from scipy.optimize import curve_fit
import numpy as np


def mean_square_error(x, y):
    return np.mean((x-y)**2)


class CurveFitModel:

    initial_params = None

    def fit(self, x, y):
        initial_params = self.initial_params
        try:
            p_opt = curve_fit(self.evaluate, x, y, initial_params)[0]
        except RuntimeError:
            print 'Optimal parameters not found for %s'%self.__class__.__name__
            p_opt = initial_params
        self.params = p_opt

    def predict(self, x, params=None):
        if params is None:
            params = self.params
        return self.evaluate(x, *params)

    def evaluate(self, x, *params):
        return NotImplementedError

    def error(self, x, y, error_function = mean_square_error):
        y_test = self.predict(x)
        return error_function(y, y_test)


class LinearModel(CurveFitModel):
    initial_params = [1, 1]
    def evaluate(self, x, A, C):
        return A*x + C


class PositiveExponentialModel(CurveFitModel):
    initial_params = [1, 1, 0]
    def evaluate(self, x, A, B, C):
        return A*np.exp(B*x) + C

class NegativeExponentialModel(CurveFitModel):
    initial_params = [1, -1, .3]
    def evaluate(self, x, A, B, C):
        return A*np.exp(B*x) + C


class LogarithmicModel(CurveFitModel):
    initial_params = [0.1, 0.1]
    def evaluate(self, x, A, C):
        return A * np.log(x) +C


class ShiftedLogarithmicModel(CurveFitModel):
    initial_params = [0.1, 0.1]
    def evaluate(self, x, A, C):
        return A*np.log(x + 1) + C


class ScaledAndShiftedLogarithmicModel(CurveFitModel):
    initial_params = [0.1, 0.1, 0.1]
    def evaluate(self, x, A, B, C):
        return A * np.log(B*x+1) + C


class HaMagicModel(CurveFitModel):
    initial_params = [1, -1, 1]
    def evaluate(self, x, A, B, C):
        return A / np.sqrt(1. + C * np.power(x, B))





registered_models = [LinearModel,
                     PositiveExponentialModel,
                     NegativeExponentialModel,
                     LogarithmicModel,
                     ShiftedLogarithmicModel,
                     ScaledAndShiftedLogarithmicModel,
                     HaMagicModel]


def best_model(x_train, y_train, x_test, y_test):
    min_error = np.inf
    best_model = None
    for model in registered_models:
        model = model()
        model.fit(x_train, y_train)
        error = model.error(x_test, y_test)
        if error < min_error:
            best_model = model
            min_error = error
    return best_model, min_error




