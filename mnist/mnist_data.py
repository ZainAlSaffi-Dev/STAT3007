"""
MNIST in the form train_run expects: flat inputs and one-hot targets.

The setup follows Liu, Michaud and Tegmark (2023), Omnigrok, arXiv:2210.01117,
Section 4 and Appendix A. They induce grokking on MNIST by training on a
random subset of 1000 of the 60000 training images and by scaling up the
initial weights. They train with mean squared error on one-hot targets. The
test set here is the full set of 10000 test images.

The images are read once with scikit-learn's fetch_openml, as the OpenML
dataset mnist_784, and cached as a compressed numpy file in data/ next to this
module. The cache is not committed. The pixels are scaled to [0, 1] and then
standardised by the mean and standard deviation of all training pixels, so
an image has a norm of about the square root of 784. The first 60000 rows of
mnist_784 are the training images and the last 10000 are the test images.
"""

import functools
from pathlib import Path

import numpy as np
import torch

CACHE = Path(__file__).resolve().parent / "data" / "mnist.npz"
N_CLASSES = 10


def _download():
    """Fetch mnist_784 from OpenML and write the cache."""
    from sklearn.datasets import fetch_openml
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True,
                        data_home=str(CACHE.parent / "openml"), parser="liac-arff")
    X = (X.astype(np.float32) / 255.0)
    y = y.astype(np.int64)
    np.savez_compressed(CACHE, X_train=X[:60000], y_train=y[:60000], X_test=X[60000:], y_test=y[60000:])


def _one_hot(labels):
    return torch.nn.functional.one_hot(torch.as_tensor(labels), N_CLASSES).float()


@functools.cache
def _arrays():
    """The stored arrays and the mean and standard deviation of all training pixels, read once per process.

    Each lookup in the NpzFile that np.load returns reads and decompresses
    that array again, so the arrays are read once here and kept.
    """
    if not CACHE.exists():
        _download()
    with np.load(CACHE) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, arrays["X_train"].mean(), arrays["X_train"].std()


def load_mnist(n_train=1000, seed=42):
    """Return ((X_train, y_train), (X_test, y_test)) as float32 tensors.

    The training set is n_train images drawn without replacement from the
    60000 training images by a generator seeded with seed. Its order is the
    order of the draw, so its first images are themselves a random draw. The
    test set is all 10000 test images in their stored order.
    """
    data, mean, std = _arrays()
    idx = np.random.default_rng(seed).choice(len(data["X_train"]), size=n_train, replace=False)
    X_train = torch.from_numpy((data["X_train"][idx] - mean) / std)
    X_test = torch.from_numpy((data["X_test"] - mean) / std)
    return (X_train, _one_hot(data["y_train"][idx])), (X_test, _one_hot(data["y_test"]))
