from typing import *

import numpy as np


__all__ = (
    # typing
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence',
    'overload', 'Any', 'TypeVar', 'Callable', 'Type', 'TYPE_CHECKING',
    'Generator', 'Reversible', 'MutableSequence',

    # custom
    'Array', 'SVDTuple', 'SVDCallable', 'SVDToInt', 'AbsColor', 'PartialColor',
    'Updater',
)

Array = np.ndarray
SVDTuple = Tuple[Array, Array, Array]
SVDCallable = Callable[..., SVDTuple]
SVDToInt = Callable[[Array, Array, Array], int]
Updater = Callable[[Iterable[Array], Iterable[Array]], Array]
AbsColor = int
PartialColor = float
