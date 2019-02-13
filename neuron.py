import numpy as np
from scipy.special import expit

#def S(x):
#    return expit(x)


#def f(w, n, x):
#    return S(w @ x.T + n)


#def j(w, n, x, y):
#    predict = f(w, n, x)
#    prop_of_ones = y * np.log(predict)
#    prop_of_zeros = (1 - y) * np.log(1 - predict)
#    return -np.mean(prop_of_ones + prop_of_zeros)


#def j_ableitung_w(w, n, x, y):
#    e = f(w, n, x) - y
#    return np.mean(x.T * e, axis=1)


#def j_ableitung_n(w, n, x, y):
#    e = f(w, n, x) - y
#    return np.mean(e)

class Neuron:

    def __init__(self, weight_count, learn_rate=0.1):
        self.w = np.zeros((1, weight_count), np.float32)
        self.b = 0.0
        self.learn_rate = learn_rate

    def predict(self, x):
        return expit(self.w @ x.T + self.b)

    def train(self, x, y):
        e = self.predict(x) - y
        dw = np.mean(x.T * e, axis=1)
        db = np.mean(e)
        self.w -= self.learn_rate * dw
        self.b -= self.learn_rate * db

    def costs(self, x, y):
        predict = self.predict(x)
        prop_of_one = y * np.log(predict)
        prop_of_zero = (1 - y) * np.log(1 - predict)
        return -np.mean(prop_of_one + prop_of_zero)
