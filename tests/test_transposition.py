from tensor_networks.transposition import transpose_bond_indices, reverse_transpose
from tests.helpers import constant_fixture, arange_from_shape

import numpy as np


arr = constant_fixture(params=[
    arange_from_shape(9, 9),
    np.random.random((3, 7, 5, 3, 2)),
    np.array([]),
    arange_from_shape(8),
])


def test_transpose_bond_indices(arr):
    assert (arr == transpose_bond_indices(transpose_bond_indices(arr))).all()


def test_reverse_transpose(arr):
    assert (arr == reverse_transpose(reverse_transpose(arr))).all()
