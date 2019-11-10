from typing import *

from numpy import ndarray

__all__ = ('List', 'Tuple', 'Iterable', 'Optional', 'Iterator',
           'ndarray',
           'SVD', 'AbsColor', 'PartialColor', )

TensorTrain = List[ndarray]
SVD = Tuple[ndarray, ndarray, ndarray]
AbsColor = int
PartialColor = float
