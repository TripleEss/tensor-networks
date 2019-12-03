from functools import partial
from itertools import accumulate

import numpy as np
from more_itertools import last

from tensor_networks.annotations import *


def contract(t1: ndarray, t2: ndarray, axes=1, **kwargs):
    return np.tensordot(t1, t2, axes=axes, **kwargs)


def tensor_accumulate(tensors: Iterable[ndarray], **kwargs) \
        -> Iterable[ndarray]:
    """
    :return:
        The arrays obtained by consecutively contracting every tensor
        over the bond indices
    """
    return accumulate(tensors, partial(contract, **kwargs))


def tensor_reduce(tensors: Iterable[ndarray], **kwargs) -> ndarray:
    """
    :return:
        The array obtained by contracting every tensor
        over the bond indices
    """
    return last(tensor_accumulate(tensors, **kwargs))


def tensor_reduce_fully(tensors: Iterable[ndarray], **kwargs) -> ndarray:
    """
    :return:
        The array obtained by contracting every tensor and the
        outer indices on the left- and right-most tensors
    """
    return tensor_reduce(tensors, **kwargs).trace(axis1=0, axis2=-1)
