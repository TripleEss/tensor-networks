from typing import *

from numpy import ndarray

__all__ = (
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence', 'overload',
    'ndarray',
    'SVD', 'AbsColor', 'PartialColor'
)

TensorTrain = List[ndarray]
SVD = Tuple[ndarray, ndarray, ndarray]
AbsColor = int
PartialColor = float
