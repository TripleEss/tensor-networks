from functools import wraps

import numpy as np

from tensor_networks.annotations import *


def as_dtype(dtype: np.dtype):
    def decorator(array_constructor: Callable[..., Array]):
        @wraps(array_constructor)
        def changed_dtype(*args, **kwargs):
            kwargs.setdefault('dtype', dtype)
            return array_constructor(*args, **kwargs)

        return changed_dtype

    return decorator


def patch_dtype(dtype: np.dtype):
    decorator = as_dtype(dtype)
    np.array = decorator(np.array)
    np.ones = decorator(np.ones)
    np.zeros = decorator(np.zeros)
