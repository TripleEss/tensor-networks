from tensor_networks.annotations import *


def reverse_transpose(arr: Array) -> Array:
    return arr.transpose(*reversed(list(range(arr.ndim))))


def transpose_bond_indices(arr: Array) -> Array:
    if arr.ndim < 2:
        return arr
    return arr.transpose(-1, *range(1, arr.ndim - 1), 0)
