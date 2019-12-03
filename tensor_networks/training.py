from itertools import islice

import numpy as np

from tensor_networks.annotations import *
from tensor_networks.contraction import contract, tensor_reduce
from tensor_networks.svd import truncated_svd


def sweep(tensor_train: TTrain,
          attachments: Sequence[ndarray],
          label_index: int = 0,
          direction: int = 1,
          svd: Callable[..., SVDTuple] = truncated_svd,
          ) -> Generator[Tuple[ndarray, ndarray], ndarray, None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param tensor_train: the train
    :param attachments:
    :param label_index: The index at which we start optimizing
    :param direction: The direction in which we start sweeping
    :param svd: The function used for singular value decomposition
    """
    assert direction in (-1, 1)
    assert len(attachments) == len(tensor_train)

    left_index, right_index = [label_index, label_index + direction][::direction]
    accumulated_left = list(tensor_train[:left_index]
                            .attach(attachments[:left_index])
                            .accumulate())
    accumulated_right = list(tensor_train[right_index + 1::-1]
                             .attach(attachments[right_index + 1:])
                             .accumulate())

    while True:
        left_index += direction
        right_index += direction
        if left_index < 0 or right_index >= len(tensor_train):
            direction *= -1
            left_index += 2 * direction
            right_index += 2 * direction

        to_optimize = contract(tensor_train[left_index], tensor_train[right_index])
        output = tensor_reduce([
            to_optimize, accumulated_right[-1], attachments[right_index],
            attachments[left_index], accumulated_left[-1]
        ])
        optimized = to_optimize + (yield to_optimize, output)

        u, s, v = svd(optimized)
        label_index, other_index = [left_index, right_index][::direction]
        tensor_train[label_index] = u
        tensor_train[other_index] = np.diag(s) @ v

        if direction == 1:
            accumulated_left.append(contract(accumulated_left[-1],
                                             tensor_train[label_index]))
            accumulated_right.pop()
        elif direction == -1:
            accumulated_right.append(np.tensordot(accumulated_right[-1],
                                                  tensor_train[label_index],
                                                  axes=(-1, -1)))
            accumulated_left.pop()


def sweep_until(*args, iterations: Optional[int] = None, **kwargs):
    # TODO: other break conditions
    return islice(sweep(*args, **kwargs), stop=iterations)


def cost(labels1: ndarray, labels2: ndarray) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2
