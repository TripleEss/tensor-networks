import math
from functools import partial
from itertools import accumulate, chain

import numpy as np

from tensor_networks.svd import truncated_svd
from utils.annotations import *


class TensorTrain(ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(TensorTrain)

    # TODO: figure out what to do with the label index (perhaps save separately and consider whenever __getitem__
    #  accesses the tensor with the label)
    def sweep(self, ) -> 'TensorTrain':
        self.reduce()
        raise NotImplementedError

    def accumulate(self) -> Iterable[ndarray]:
        """
        :return: The arrays obtained by consecutively contracting every tensor
        """
        return accumulate(self, partial(np.tensordot, axes=1))

    def reduce(self) -> ndarray:
        """
        :return:
            The array obtained by contracting every tensor and the
            mock indices on the left- and right-most tensors
        """
        return list(self.accumulate())[-1].trace(axis1=0, axis2=-1)

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


class AttachedTensorTrain(Sequence[Tuple[ndarray, ndarray]]):
    def __init__(self, train: TensorTrain, attachment: Sequence[ndarray], is_reversed=False):
        """
        :param train: TensorTrain
        :param attachment:
            A list of arrays.
            Each element of self will be contracted with the corresponding element of others
        :param is_reversed:
            Whether the tensor train and attachment are in reverse order
        """
        assert len(train) == len(attachment)
        self.train = train
        self.attachment = attachment
        self.is_reversed = is_reversed

    def accumulate(self, attachment_index: int = 0) -> Iterable[ndarray]:
        """
        :param attachment_index:
            The index each element of others will get contracted over
        :return:
            The array obtained by consecutively contracting self while contracting each
            intermediate result with the corresponding element of others
        """
        i0, i1 = (0, 1)
        if self.is_reversed:
            i0, i1 = ~i0, ~i1

        def reduction_step(result: ndarray, new: Tuple[ndarray, ndarray]) -> ndarray:
            return np.tensordot(
                np.tensordot(result, new[0], axes=([i1], [i0])),
                new[1],
                axes=([i1], [attachment_index])
            )

        return accumulate(
            # chain start value with the rest of self
            chain(np.tensordot(*self[0], axes=([i1], [attachment_index])),
                  self[1:]),
            reduction_step,
        )
        # TODO: what to do with the mock indices

    @overload
    def __getitem__(self, item: int) -> Tuple[ndarray, ndarray]:
        ...

    @overload
    def __getitem__(self, item: slice) -> 'AttachedTensorTrain':
        ...

    def __getitem__(self, item):
        if isinstance(item, slice):
            return type(self)(
                self.train[item],
                self.attachment[item],
                is_reversed=(self.is_reversed if slice.step > 0
                             else not self.is_reversed)
            )
        return self.train[item], self.attachment[item]

    def __reversed__(self) -> 'AttachedTensorTrain':
        return self[::-1]

    def __len__(self) -> int:
        return len(self.train)

    def __iter__(self) -> Iterable[Tuple[ndarray, ndarray]]:
        return zip(self.train, self.attachment)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(train={self.train!r}, attachment=' \
               f'{self.attachment!r}, is_reversed={self.is_reversed!r})'

    def __str__(self) -> str:
        return f'{type(self).__name__}(train={self.train}, attachment=' \
               f'{self.attachment}, is_reversed={self.is_reversed})'
