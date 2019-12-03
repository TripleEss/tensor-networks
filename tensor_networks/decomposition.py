import math

import numpy as np

from tensor_networks.annotations import *
from tensor_networks.svd import truncated_svd


_T = TypeVar('_T')


def decompose(cls: Callable[[List[ndarray]], _T], tensor: ndarray, d: int,
              svd: Callable[..., SVDTuple] = truncated_svd,
              **svd_kwargs) -> _T:
    """
    Decompose a tensor into the tensor train format using SVD

    :param cls: A class that wraps the tensor train format
    :param tensor: The tensor to decompose
    :param d: The dimension of the bond indices
    :param svd: The function used for singular value decomposition
    :param svd_kwargs:
        Any keyworded arguments are passed through to the svd function
        (for convenience)
    :return: The tensor in tensor train format
    """
    # Amount of elements in the tensor: $d^N$ (= tensor.size)
    # $\Longleftrightarrow N = log_d(d^N)$
    n = int(math.log(tensor.size, d))
    # Add a mock index on the left for the first iteration
    t = tensor.reshape(1, tensor.size)
    train = []
    for i in range(1, n):
        # Reshape the tensor into a matrix (to calculate the SVD)
        t.shape = (d * t.shape[0], d ** (n - i))
        # Split the tensor using Singular Value Decomposition (SVD)
        u, s, v = svd(t, **svd_kwargs)
        # Split the first index of the matrix u
        u.shape = (u.shape[0] // d, d, u.shape[1])
        # u is part of the tensor train
        train.append(u)
        # Continue, using the contraction of s and v as the remaining tensor
        t = np.diag(s) @ v
    # The remaining matrix is the right-most tensor in the tensor train
    # and gets a mock index on the right for consistency
    t.shape = (*t.shape, 1)
    train.append(t)
    return cls(train)
