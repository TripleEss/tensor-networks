from typing import *

from tensor_networks.patched_numpy import np


__all__ = (
    # typing
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence',
    'overload', 'Any', 'TypeVar', 'Callable', 'Type', 'TYPE_CHECKING',
    'Generator', 'Reversible', 'MutableSequence', 'NamedTuple',

    # custom
    'Array', 'SVDTuple', 'SVDCallable', 'SVDToInt', 'AbsColor', 'PartialColor',
    'Updater',
)

Array = np.ndarray
SVDTuple = Tuple[Array, Array, Array]
SVDCallable = Callable[..., SVDTuple]
SVDToInt = Callable[[Array, Array, Array], int]
Updater = Callable[[Array, Array, Array], Array]
AbsColor = int
PartialColor = float
