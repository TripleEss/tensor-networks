import numpy as np

from tensor_networks.annotations import *


_T = TypeVar('_T', bound=ndarray)


def attach(tensor_train: _T, attachments: Iterable[ndarray]) -> _T:
    """
    :param tensor_train:
    :param attachments:
        Tensors to be contracted with
    :return:
        Every core contracted with its respective attachment
    """
    train = []
    for core, attached in zip(tensor_train, attachments):
        train.append(np.tensordot(core, attached, axes=(1, 0)))
    return type(tensor_train)(train)
