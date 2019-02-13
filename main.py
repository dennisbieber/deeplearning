import numpy as np
from neuron import Neuron
import mnistreadlib as mr

xs = mr.read_mnist_images("train-images-idx3-ubyte.gz").reshape(-1, 784)
ys = (mr.read_mnist_labels("train-labels-idx1-ubyte.gz") == 8).astype(np.float32)

xtest = mr.read_mnist_images("t10k-images-idx3-ubyte.gz").reshape(-1, 784)
ytest = mr.read_mnist_labels("t10k-labels-idx1-ubyte.gz") == 8

n = Neuron(784, 0.00001)

for k in range(0, 60000, 1000):

    n.train(xs[k:k + 1000, :], ys[k:k + 1000])

    cost = n.costs(xs, ys)
    pred = np.mean(((n.predict(xtest) > 0.5) == ytest).astype(np.float32))
    print(k + 1000, "Vorhersagekraft:", pred, ", Kosten:", cost)




