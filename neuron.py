import numpy as np
from scipy.special import expit

class Neuron:

    def __init__(self, weight_count, output_count):
        self.w = np.zeros((output_count, weight_count))
        self.b = np.zeros(output_count)

    def predict(self, x):
        return expit((self.w @ x.T).T + self.b).T

    def train(self, x, y, learn_rate=0.1):
        e = self.predict(x) - y
        dw = (x.T @ e.T / x.shape[0]).T
        db = np.mean(e, axis=1)
        self.w -= learn_rate * dw
        self.b -= learn_rate * db

    def costs(self, x, y):
        predict = self.predict(x)
        prop_of_one = y * np.log(predict)
        prop_of_zero = (1 - y) * np.log(1 - predict)
        return -np.mean(prop_of_one + prop_of_zero, axis=1)
