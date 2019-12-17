from functools import partial
from itertools import accumulate

import numpy as np
from more_itertools import last

from tensor_networks.annotations import *


def contractions(*tensors: ndarray, axes=1, **kwargs) -> Iterator[ndarray]:
    """
    :return: The arrays obtained by consecutively contracting every tensor
    """
    return accumulate(tensors, partial(np.tensordot, axes=axes, **kwargs))


def contract(*tensors: ndarray, **kwargs) -> ndarray:
    """
    :return: The array obtained by contracting every tensor
    """
    return last(contractions(*tensors, **kwargs))
