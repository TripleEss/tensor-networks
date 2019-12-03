from __future__ import annotations

import numpy as np

from tensor_networks import (classification, contraction, decomposition,
                             training)
from tensor_networks.annotations import *
from tensor_networks.transposition import transpose_bond_indices


class TensorTrain(ndarray):
    # these functions are added as methods for convenience
    attach = classification.attach
    accumulate = contraction.tensor_accumulate
    reduce = contraction.tensor_reduce
    reduce_fully = contraction.tensor_reduce_fully
    decompose = classmethod(decomposition.decompose)
    sweep = training.sweep

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
