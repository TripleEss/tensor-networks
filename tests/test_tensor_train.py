import numpy as np
import pytest
from pytest import approx

from tensor_networks.misc import reverse_transpose
from tensor_networks.tensor_train import TensorTrain
from tests.helpers import constant_fixture


data_arrays, data_ds = zip(
    (np.arange(64), 2),
    (np.arange(4), 2),
)
data_ids = [str(x) for x in range(len(data_arrays))]

arr = constant_fixture(params=data_arrays, ids=data_ids)
d = constant_fixture(params=data_ds, ids=data_ids)
chi = constant_fixture(params=[2])


@pytest.fixture(ids=data_ids)
def tt(arr: np.ndarray, d: int, chi: int):
    return TensorTrain.decompose(arr, d=d, chi=chi)


def test_shape(tt, d, chi):
    for s in tt.shape:
        # general shape
        assert len(s) == 3
        # physical indices
        assert s[1] == d
    # bond indices
    for s in tt.shape[1:]:
        assert s[0] == chi
    for s in tt.shape[:-1]:
        assert s[-1] == chi
    # mock bond indices
    assert tt.shape[0][0] == 1
    assert tt.shape[-1][-1] == 1


@pytest.fixture(ids=data_ids)
def accumulated(tt):
    return list(tt.accumulate())


def test_accumulate(tt):
    for l, t in zip(range(3, len(tt)), tt.accumulate()):
        assert len(t.shape) == l


@pytest.fixture(ids=data_ids)
def reduced(tt):
    return tt.reduce()


def test_reassemble(arr, d, reduced):
    assert np.prod(reduced.shape) == arr.shape
    for i in reduced.shape:
        assert i == d
    assert reduced.flatten() == approx(arr.flatten())


def test_reversed(tt, reduced):
    rev = tt[::-1]
    for x, y in zip(reversed(tt), rev):
        assert (x == y).all()
    rev_reassembled = rev.reduce()
    assert reverse_transpose(rev_reassembled) == approx(reduced)
