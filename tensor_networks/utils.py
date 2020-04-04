from tensor_networks.annotations import *

import numpy as np


ONE_TENSOR = np.array([1])

_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')

def get(l: List[_T1], index: int, default: _T2 = None) -> Union[_T1, _T2]:
    return l[index] if index < len(l) and abs(index) <= len(l) else default


def get_last(l: List[_T1], default: _T2 = None) -> Union[_T1, _T2]:
    return get(l, index=-1, default=default)

def get_last_or_one_tensor(l: List[_T1]) -> Union[_T1, Array]:
    return get_last(l, default=ONE_TENSOR)

def identity(x: _T1) -> _T1:
    return x
