from typing import *

from numpy import ndarray


__all__ = (
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence',
    'overload', 'Any', 'TypeVar', 'Callable',
    'ndarray',
    'SVD', 'AbsColor', 'PartialColor'
)

SVD = Tuple[ndarray, ndarray, ndarray]
AbsColor = int
PartialColor = float
