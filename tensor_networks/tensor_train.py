from __future__ import annotations

import numpy as np

from tensor_networks import (classification, contraction, decomposition,
                             training)
from tensor_networks.annotations import *
from tensor_networks.transposition import transpose_bond_indices


class TensorTrain(ndarray, Sequence[ndarray]):
    attach = classification.attach
    decompose = classmethod(decomposition.decompose)

    def accumulate(self, **kwargs) -> Iterable[ndarray]:
        return contraction.contractions(*self, **kwargs)

    def reduce(self, **kwargs) -> ndarray:
        return contraction.contract(*self, **kwargs)

    def reduce_fully(self, **kwargs) -> ndarray:
        """
        :return:
            The array obtained by contracting every tensor and the
            outer indices on the left- and right-most tensors
        """
        return self.reduce(**kwargs).trace(axis1=0, axis2=-1)

    def __new__(cls, cores, **kwargs):
        if isinstance(cores, ndarray):
            arr = cores
        else:
            arr = np.empty([len(cores)], dtype=object, **kwargs)
            arr[:] = cores
        return arr.view(cls)

    @property  # type: ignore[misc]
    def shape(self):
        return [t.shape for t in self]

    @overload
    def __getitem__(self, item: int) -> ndarray:
        ...

    @overload
    def __getitem__(self, item: slice) -> TensorTrain:
        ...

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice) and item.step is not None and item.step < 0:
            # transpose bond indices if the train gets reversed
            return type(self)([transpose_bond_indices(arr) for arr in result])
        return result

    def __reversed__(self) -> Iterator[ndarray]:
        return iter(self[::-1])

    def __str__(self) -> str:
        return '[' + ' '.join(str(t) for t in self) + ']'

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self})'
