from typing import *

from numpy import ndarray


__all__ = (
    # typing
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence',
    'overload', 'Any', 'TypeVar', 'Callable', 'Type', 'TYPE_CHECKING',
    'Generator', 'Reversible', 'MutableSequence',

    # external
    'ndarray',

    # internal
    'SVDTuple', 'SVDCallable', 'SVDToInt', 'AbsColor', 'PartialColor',
    'TTrain', 'Updater',
)

SVDTuple = Tuple[ndarray, ndarray, ndarray]
SVDCallable = Callable[..., SVDTuple]
SVDToInt = Callable[[ndarray, ndarray, ndarray], int]
Updater = Callable[[Iterable[ndarray], Iterable[ndarray]], ndarray]
AbsColor = int
PartialColor = float
if TYPE_CHECKING:
    from tensor_networks.tensor_train import TensorTrain as TTrain
else:
    TTrain = None
