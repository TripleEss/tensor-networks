import math
from functools import reduce, partial

import numpy as np

from my_types import *
from svd import truncated_svd


class TensorTrain(Tuple[ndarray, ...]):
    # TODO: figure out what to do with the label index
    def fold(self) -> ndarray:
        """
        :return: The array obtained by contracting every index
        """
        return reduce(
            partial(np.tensordot, axes=1),
            self[1:],
            self[0]
        ).trace(axis1=0, axis2=-1)

    def fold_zip(self, others: List[ndarray], other_index: int = 0,
                 zip_slice: slice = slice(None, None, 1)) -> ndarray:
        """
        :param others:
            A list of arrays.
            Each element of self will be contracted with the corresponding element of others
        :param other_index:
            The index each element of others will get contracted over
        :param zip_slice:
            A slice object which will be applied to the zip of self and others
            (Use slice(..., ..., -1) to fold from right to left)
        :return:
            The array obtained by folding self while contracting each
            intermediate result with the corresponding element of others
        """
        assert len(self) == len(others)
        i0, i1 = (0, 1)
        if zip_slice.step < 0:
            i0, i1 = ~i0, ~i1
            zip_slice = slice(zip_slice.start, zip_slice.stop - 1, zip_slice.step)
        else:
            zip_slice = slice(zip_slice.start + 1, zip_slice.stop, zip_slice.step)

        def reduction_step(new: Tuple[ndarray, ndarray], result: ndarray) -> ndarray:
            return np.tensordot(
                np.tensordot(
                    result,
                    new[0],
                    axes=([i1], [i0])
                ),
                new[1],
                axes=([i1], [other_index])
            )

        return reduce(reduction_step,
                      zip(self[zip_slice], others[zip_slice]),
                      np.tensordot(self[0], others[0], axes=([i1], [other_index])))

    @classmethod
    def decompose(cls, tensor: ndarray, d: int, chi: Optional[int] = None) -> 'TensorTrain':
        """
        Decompose a tensor into the tensor train format using SVD

        :param tensor: The tensor to decompose
        :param chi: How many elements to keep when splitting the tensor using SVD
        :param d: The dimension of the spatial indices
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
            u, s, v = truncated_svd(t, chi)
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

    def __repr__(self):
        type(self).__name__ + super().__repr__()

    __str__ = __repr__
