from __future__ import annotations

import math

import numpy as np

from tensor_networks import contraction, training
from tensor_networks.annotations import *
from tensor_networks.svd import truncated_svd
from tensor_networks.transposition import transpose_bond_indices


class TensorTrain(Sequence[ndarray]):
    """
    This class represents a tensor train decomposition.
    It acts as an array of its cores.
    Every core's indices are assumed to have the following meaning:
        0: left bond index
        1: physical index
        -1: right bond index
    Additionally, there is always one core with the additional index:
        2: label index
    """

    sweep = training.sweep
    sweep_until = training.sweep_until

    def __init__(self, cores: MutableSequence[ndarray]):
        self.cores = cores

    @classmethod
    def decompose(cls, tensor: ndarray, d: int,
                  svd: SVDCallable = truncated_svd,
                  **svd_kwargs) -> TensorTrain:
        """
        Decompose a tensor into the tensor train format using SVD

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

    def contractions(self, keep_mock_index=True, **kwargs) -> Iterable[ndarray]:
        if not keep_mock_index and len(self) > 0 and self[0].shape[0] == 1:
            first = self[0].reshape(self[0].shape[1:])
            cores = TensorTrain([first, *self[1:]])
        else:
            cores = self
        return contraction.contractions(*cores, **kwargs)

    def contract(self, fully=False, **kwargs) -> ndarray:
        """
        :param fully: Whether to contract the outer indices
        :return: The array obtained by contracting every core
        """
        contracted = contraction.contract(*self, **kwargs)
        if fully:
            contracted = contracted.trace(axis1=0, axis2=-1)
        return contracted

    def attach(self, attachments: Iterable[ndarray]) -> TensorTrain:
        """
        :param attachments: Tensors to be contracted with
        :return: Every core contracted with its respective attachment
        """
        train = []
        for core, attached in zip(self, attachments):
            train.append(contraction.contract(core, attached, axes=(1, 0)))
        return type(self)(train)

    @property
    def shape(self):
        return [t.shape for t in self]

    @overload
    def __getitem__(self, item: int) -> ndarray:
        ...

    @overload
    def __getitem__(self, item: slice) -> TensorTrain:
        ...

    def __getitem__(self, item):
        value = self.cores[item]
        if isinstance(item, slice):
            if item.step is not None and item.step < 0:
                # transpose bond indices if the train gets reversed
                value = [transpose_bond_indices(arr) for arr in value]
            return type(self)(value)
        return value

    def __setitem__(self, key: int, value: ndarray):
        self.cores[key] = value

    def __reversed__(self) -> Iterator[ndarray]:
        return iter(self[::-1])

    def __len__(self) -> int:
        return len(self.cores)

    def __iter__(self) -> Iterator[ndarray]:
        return iter(self.cores)

    def __str__(self) -> str:
        return str(self.cores)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self})'
