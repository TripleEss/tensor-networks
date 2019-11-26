from functools import partial
from itertools import accumulate

import numpy as np
from more_itertools import last  # type: ignore[import]

from tensor_networks.utils.tensors import transpose_bond_indices
from tensor_networks.utils.annotations import *


class TensorTrain(ndarray):
    def __new__(cls, cores, **kwargs):
        if isinstance(cores, ndarray):
            arr = cores
        else:
            arr = np.empty([len(cores)], dtype=object, **kwargs)
            arr[:] = cores
        return arr.view(cls)

    def accumulate(self, **kwargs) -> Iterable[ndarray]:
        """
        :return:
            The arrays obtained by consecutively contracting every tensor
            over the bond indices
        """
        return accumulate(self, partial(np.tensordot, axes=1, **kwargs))

    def reduce(self, **kwargs) -> ndarray:
        """
        :return:
            The array obtained by contracting every tensor
            over the bond indices
        """
        return last(self.accumulate(**kwargs))

    def reduce_fully(self, **kwargs) -> ndarray:
        """
        :return:
            The array obtained by contracting every tensor and the
            outer indices on the left- and right-most tensors
        """
        return self.reduce(**kwargs).trace(axis1=0, axis2=-1)

    def sweep(self, optimizer: Callable[[ndarray], ndarray],
              svd: Callable[..., SVD],
              label_index: int = 0, direction: int = 1) -> None:
        """
        Sweep back and forth through the train and optimize the cores

        :param optimizer: The function which actually optimizes the cores
        :param svd: The function used for singular value decomposition
        :param label_index: The index at which we start optimizing
        :param direction: The direction in which we start sweeping
        """
        assert direction in (-1, 1)
        next_index = label_index + direction
        accumulated_left = list(self[:min(label_index, next_index)]
                                .accumulate())
        accumulated_right = list(self[max(label_index, next_index) + 1::-1]
                                 .accumulate())
        while True:
            next_index = label_index + direction
            if next_index > len(self) - 1:
                direction *= -1
                next_index = label_index + direction

            left_index = min(label_index, next_index)
            right_index = max(label_index, next_index)
            to_optimize = TensorTrain([
                accumulated_left[-1],
                self[left_index], self[right_index],
                accumulated_right[-1]
            ]).reduce()
            optimized = optimizer(to_optimize)
            u, s, v = svd(optimized)
            self[label_index] = u
            self[next_index] = np.diag(s) @ v

            if direction == 1:
                accumulated_left.append(np.tensordot(
                    accumulated_left[-1],
                    self[label_index],
                    axes=1
                ))
                accumulated_right.pop()
            elif direction == -1:
                accumulated_right.append(np.tensordot(
                    accumulated_right[-1],
                    self[label_index],
                    axes=(-1, -1)
                ))
                accumulated_left.pop()
            # point index to the tensor which now has the label index
            label_index = next_index

    def attach(self, attachments: Iterable[ndarray],
               attachment_index: int = 0) -> 'TensorTrain':
        """
        :param attachments:
        :param attachment_index:
            The index each element of the attachments will get contracted over
        :return:
            Every core contracted with its respective attachment
        """
        train = []
        for core, attached in zip(self, attachments):
            train.append(np.tensordot(core, attached,
                                      axes=(1, attachment_index)))
        return type(self)(train)

    @property  # type: ignore[misc]
    def shape(self):
        return [t.shape for t in self]

    @overload
    def __getitem__(self, item: int) -> ndarray:
        ...

    @overload
    def __getitem__(self, item: slice) -> 'TensorTrain':
        ...

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice) and item.step is not None and item.step < 0:
            # transpose bond indices if the train gets reversed
            return type(self)([transpose_bond_indices(arr) for arr in result])
        return result

    def __reversed__(self) -> Iterator[ndarray]:
        return iter(self[::-1])

    def __str__(self):
        return '[' + ' '.join(str(t) for t in self) + ']'

    def __repr__(self):
        return f'{type(self).__name__}({self})'
