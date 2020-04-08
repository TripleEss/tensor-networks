"""
(Trickery to allow automatic usage of either CuPy or Numpy depending on
whether CuPy is available.)
TODO: this is not implemented yet since there are more differences than \
 expected: https://docs-cupy.chainer.org/en/stable/reference/difference.html

The automatically detected module is copied and exposed via the np and numpy
variables.

Additionally one can set GLOBAL_NUMERIC_DATA_TYPE to e.g. np.float32
to use this as the default data type for NumPy arrays.
This only modifies the behaviour of the local copy of NumPy defined in here.
"""
import functools
import importlib
import logging
import typing

if typing.TYPE_CHECKING:
    import numpy as _backend_module
    np = numpy = _backend_module
else:
    # try:
    #     import cupy as _backend_module
    #     _backend_spec = importlib.util.find_spec('cupy')
    # except ImportError:
    #     logging.info("Either CuPy or CUDA could not be found. "
    #                  "Using the regular NumPy library instead.")
    import numpy as _backend_module
    _backend_spec = importlib.util.find_spec('numpy')
    # end except
    np = numpy = importlib.util.module_from_spec(_backend_spec)
    _backend_spec.loader.exec_module(np)


GLOBAL_NUMERIC_DATA_TYPE: typing.Optional[np.dtype] = None

def _as_global_dtype(array_constructor: typing.Callable[..., np.ndarray]):
    @functools.wraps(array_constructor)
    def changed_dtype(*args, **kwargs):
        if GLOBAL_NUMERIC_DATA_TYPE is not None and len(args) < 2:
            kwargs.setdefault('dtype', GLOBAL_NUMERIC_DATA_TYPE)
        return array_constructor(*args, **kwargs)
    return changed_dtype


np.array = _as_global_dtype(_backend_module.array)
np.ones = _as_global_dtype(_backend_module.ones)
np.zeros = _as_global_dtype(_backend_module.zeros)
np.full = _as_global_dtype(_backend_module.full)
