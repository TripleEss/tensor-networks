import math
import logging
from typing import List

import numpy as np

from setup_logging import setup_logging
from svd import truncated_svd


def decompose(tensor: np.ndarray, chi: int, d: int = 2) -> List[np.ndarray]:
    """
    Decompose a tensor into the tensor train format using SVD

    :param tensor: The tensor to decompose (has to be a vector)
    :param chi: How many elements to keep when splitting the tensor using SVD
    :param d: The dimension of the spatial indices
    :return: The tensor in tensor train format
    """
    assert tensor.ndim == 1
    # Amount of elements in the tensor: $d^N$ (= tensor.shape[0])
    # $\Longleftrightarrow N = log_d(d^N)$
    n = int(math.log(tensor.shape[0], d))
    tensor_train = []
    # Add a second mock index on the left for the first iteration
    tensor.shape = (1, *tensor.shape)
    for i in range(1, n):
        # Reshape the tensor into a matrix (to calculate the SVD)
        tensor.shape = (d * tensor.shape[0], d ** (n - i))
        # Split the tensor using Singular Value Decomposition (SVD)
        u, s, v = truncated_svd(tensor, chi)
        # Split the first index of the matrix u
        u.shape = (u.shape[0] // d, d, u.shape[1])
        # u is part of the tensor train
        tensor_train.append(u)
        # Continue, using the contraction of s and v as the remaining tensor
        tensor = np.diag(s) @ v
    # The remaining matrix is the right-most tensor in the tensor train
    # and gets a mock index on the right for consistency
    tensor.shape = (*tensor.shape, 1)
    tensor_train.append(tensor)
    return tensor_train


def phi_left(f: List[np.ndarray], b: List[np.ndarray], l: int) -> np.ndarray:
    phi = np.tensordot(f[0], b[0], axes=([1], [0]))
    for f_i, b_i in zip(f[1:l], b[1:l]):
        phi = np.tensordot(phi, f_i, axes=([-1], [0]))
        phi = np.tensordot(phi, b_i, axes=([-1], [0]))
    return phi


def phi_right(f: List[np.ndarray], b: List[np.ndarray], l: int) -> np.ndarray:
    phi = np.tensordot(f[-1], b[-1], axes=([0], [0]))
    for f_i, b_i in zip(f[l + 2:-1:-1], b[l + 2:-1:-1]):
        phi = np.tensordot(phi, f_i, axes=([0], [-1]))
        phi = np.tensordot(phi, b_i, axes=([0], [0]))
    return phi


# %% Testing
if __name__ == '__main__':
    setup_logging(logging.DEBUG)
    print('--- Testing tensor decomposition ---')
    t = np.arange(64)
    print(f"Tensor: {t}")
    ttrain = decompose(t, chi=4, d=2)
    print(f"Shape of decomposed tensor train: {[x.shape for x in ttrain]}")
    print(f"Decomposed tensor train:\n{ttrain}")
    ttest = ttrain[0]
    for x in ttrain[1:]:
        ttest = np.tensordot(ttest, x, axes=1)
    print(f"Reassembled tensor train:\n{ttest}")
