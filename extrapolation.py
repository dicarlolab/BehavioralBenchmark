__author__ = 'ardila'
from scipy.optimize import curve_fit
import numpy as np

def mean_square_error(x, y):
    return np.mean((x-y)**2)

class CurveFitModel:

    initial_params = None

    def fit(self, x, y):
        initial_params = self.initial_params
        p_opt = curve_fit(self.evaluate, x, y, initial_params)[0]
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

    def evaluate(self, x, a, b):
        return a*x + b




class ExponentialModel(CurveFitModel):

    initial_params = [1, 1, 1]

    def evaluate(self, x, A, B, C):
        return A*np.exp(B*x) + C



registered_models = [LinearModel, ExponentialModel]


def best_model(x_train, y_train, x_test, y_test):
    min_error = np.inf
    best_model = None
    for model in registered_models:
        model = model()
        model.fit(x_train, y_train)
        error = model.error(x_test, y_test)
        if error<min_error:
            best_model = model
    return best_model, error




