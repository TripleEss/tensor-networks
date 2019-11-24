import pytest


def constant_fixture(*args, **kwargs):
    return pytest.fixture(*args, **kwargs)(lambda request: request.param)
