import numpy as np
import pytest

from tensor_networks.svd import truncated_svd


@pytest.mark.parametrize('arr', [
    np.arange(50).reshape(10, 5),
    np.arange(1).reshape(1, 1),
])
@pytest.mark.parametrize('chi', [None, 1, 2, 3, 4, 10, 100])
def test_truncated_svd(arr, chi):
    v, s, h = truncated_svd(arr, chi)
    # TODO
