import gzip
import numpy as np

def read_mnist_images(name):
    with gzip.open(name, mode="rb") as f:
        return np.frombuffer(f.read(), offset=16, dtype=np.uint8)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)


def read_mnist_labels(name):
    with gzip.open(name, mode="rb") as f:
        return np.frombuffer(f.read(), offset=8, dtype=np.uint8)
