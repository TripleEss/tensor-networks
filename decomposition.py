import math
import logging
from typing import List

import numpy as np

from setup_logging import setup_logging
from svd import truncated_svd


def decompose(tensor: np.ndarray, chi: int, d: int = 2) -> List[np.ndarray]:
    """
    Decompose a tensor into the tensor train format using SVD

    :param tensor: The tensor to decompose
    :param chi: How many elements to keep when splitting the tensor using SVD
    :param d: The dimension of the spatial indices
    :return: The tensor in tensor train format
    """
    # Amount of elements in the tensor: $d^N$ (= tensor.shape[0])
    # $\Longleftrightarrow N = log_d(d^N)$
    n = int(math.log(tensor.size, d))
    # Add a mock index on the left for the first iteration
    t = tensor.reshape(1, tensor.size)
    tensor_train = []
    for i in range(1, n):
        # Reshape the tensor into a matrix (to calculate the SVD)
        t.shape = (d * t.shape[0], d ** (n - i))
        # Split the tensor using Singular Value Decomposition (SVD)
        u, s, v = truncated_svd(t, chi)
        # Split the first index of the matrix u
        u.shape = (u.shape[0] // d, d, u.shape[1])
        # u is part of the tensor train
        tensor_train.append(u)
        # Continue, using the contraction of s and v as the remaining tensor
        t = np.diag(s) @ v
    # The remaining matrix is the right-most tensor in the tensor train
    # and gets a mock index on the right for consistency
    t.shape = (*t.shape, 1)
    tensor_train.append(t)
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
    setup_logging()
    logging.debug('--- Testing tensor decomposition ---')
    t = np.arange(64)
    logging.debug(f"Original tensor's shape: {t.shape}")
    ttrain = decompose(t, chi=4, d=2)
    logging.debug(f"Shape of decomposed tensor train: {[x.shape for x in ttrain]}")
    ttest = ttrain[0]
    for x in ttrain[1:]:
        ttest = np.tensordot(ttest, x, axes=1)
    ttest.shape = ttest.shape[1:-1]
    logging.debug(f"Shape of the reassembled tensor train: {ttest.shape}")
    diff = abs(t.flatten() - ttest.flatten()).reshape(ttest.shape)
    logging.debug(f"Maximum difference between the original and the reassembled tensor: {diff.max()}")
