import numpy as np
from scipy.misc import imread
from sklearn.preprocessing import OneHotEncoder

import mnistreadlib as mr
from neuron import Neuron

xs = mr.read_mnist_images("train-images-idx3-ubyte.gz").reshape(-1, 784)
ys = mr.read_mnist_labels("train-labels-idx1-ubyte.gz")
ys = OneHotEncoder(categories='auto').fit_transform(ys.reshape(-1, 1)).toarray().T

image = 255. - imread("9.png").reshape(1, -1)

n = Neuron(784, ys.shape[0], 0.00001)

for k in range(0, 50):

    n.train(xs, ys)

    pred = n.predict(image)
    num = np.argmax(pred, axis=0)
    print(k + 1, num, "(" + str(pred.reshape(1, -1)[0][num]) + ")")
