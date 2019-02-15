import numpy as np
from sklearn.preprocessing import OneHotEncoder

import mnistreadlib as mr
from neuron import Neuron

xs = mr.read_mnist_images("train-images-idx3-ubyte.gz").reshape(-1, 784)
ys = mr.read_mnist_labels("train-labels-idx1-ubyte.gz")
ys = OneHotEncoder().fit_transform(ys.reshape(-1, 1)).toarray().T

xtest = mr.read_mnist_images("t10k-images-idx3-ubyte.gz").reshape(-1, 784)
ytest = mr.read_mnist_labels("t10k-labels-idx1-ubyte.gz")

n = Neuron(784, 10, 0.00001)

for k in range(0, 1):
    n.train(xs, ys)

    cost = n.costs(xs, ys)
    pred = np.mean(((n.predict(xtest) > 0.5) == ytest).astype(np.float32))
    print(k + 1000, "Vorhersagekraft:", pred, ", Kosten:", cost)




