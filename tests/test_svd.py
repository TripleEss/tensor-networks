import numpy as np
import pytest
from pytest import approx

from tensor_networks.svd import truncated_svd


@pytest.mark.parametrize('arr', [
    np.arange(50).reshape(10, 5),
    np.arange(1).reshape(1, 1),
])
@pytest.mark.parametrize('max_chi', [None, 1, 2, 3, 4, 10, 100])
def test_truncated_svd(arr, max_chi):
    u, s, v = truncated_svd(arr, max_chi=max_chi)
    if max_chi != 1:
        test_arr = u @ np.diag(s) @ v
        assert test_arr == approx(arr)
