from tensor_networks.annotations import *


def reverse_transpose(arr: ndarray) -> ndarray:
    return arr.transpose(*reversed(list(range(arr.ndim))))


def transpose_bond_indices(arr: ndarray) -> ndarray:
    if arr.ndim < 2:
        return arr
    return arr.transpose(-1, *range(1, arr.ndim - 1), 0)
