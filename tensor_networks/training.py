from itertools import count

import numpy as np

from tensor_networks.annotations import *
from tensor_networks.contraction import contract
from tensor_networks.inputs import Input
from tensor_networks.svd import truncated_svd


def pairwise_slices(lower_bound, upper_bound: int, start: int = 0,
                    direction: int = 1) -> Generator[slice, None, None]:
    # TODO python negative step weirdness
    i1, i2 = sorted((start, start + 2 * direction))
    assert lower_bound <= i1 < i2 <= upper_bound
    while True:
        yield slice(i1, i2, direction)
        i1 += direction
        i2 += direction
        if i1 < lower_bound or i2 > upper_bound:
            direction *= -1
            i1 += 2 * direction
            i2 += 2 * direction


def sweep(tensor_train: TTrain,
          input_: Input,
          label_index: int = 0,
          direction: int = 1,
          svd: SVDCallable = truncated_svd,
          ) -> Generator[Tuple[ndarray, ndarray], ndarray, None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param tensor_train: The train to optimize
    :param input_: TODO
    :param label_index: The index at which we start optimizing
    :param direction: The direction in which we start sweeping
    :param svd: The function used for singular value decomposition
    """
    assert direction in (-1, 1)
    assert len(input_) == len(tensor_train)

    left_i, right_i = sorted((label_index, label_index + direction))
    accumulated_left = list(tensor_train[:left_i]
                            .attach(input_[:left_i])
                            .accumulate())
    accumulated_right = list(tensor_train[:right_i:-1]
                             .attach(input_[right_i + 1:])
                             .accumulate())

    for slc in pairwise_slices(0, len(tensor_train), start=label_index):
        direction = slc.step
        label_core, other_core = tensor_train[slc]
        label_input, other_input = input_[slc]
        label_accum, other_accum = (accumulated_left,
                                    accumulated_right)[::direction]

        to_optimize = contract(label_core, other_core)
        # TODO where is the label index
        local_input = contract(label_input, other_input, axes=0)
        if label_accum:
            local_input = contract(label_accum[-1], local_input, axes=0)
        else:
            local_input.shape = (1,) + local_input.shape
        if other_accum:
            local_input = contract(local_input, other_accum[-1], axes=0)
        else:
            local_input.shape = local_input.shape + (1,)
        output = contract(
            to_optimize, local_input,
            axes=(list(range(len(to_optimize.shape))),
                  (list(range(len(local_input.shape)))))
        )
        optimized = to_optimize + (yield output, local_input)

        u, s, v = svd(optimized)
        label_core[:], other_core[:] = u, np.diag(s) @ v

        label_accum.append(contract(*(label_accum[-1], label_core)[::direction]))
        other_accum.pop()


def sweep_simultaneously(
        tensor_train: TTrain,
        inputs: Iterable[Input],
        **kwargs,
) -> Generator[Generator[ndarray, ndarray, None], None, None]:
    isweeps = [sweep(tensor_train, inp, **kwargs) for inp in inputs]
    outputs = [next(isweep) for isweep in isweeps]
    while True:
        def optimizations():
            for i, isweep in isweeps:
                update = yield outputs[i]
                outputs[i] = isweep.send(update)

        yield optimizations()


def sweep_simultaneously_until(tensor_train: TTrain,
                               inputs: Sequence[Input],
                               iterations: Optional[int] = None,
                               **kwargs) -> None:
    # TODO: other break conditions
    counter = range(iterations) if iterations else count()
    isweep = sweep_simultaneously(tensor_train, inputs, **kwargs)
    outputs = list(next(isweep))
    for optimizations, _ in zip(isweep, counter):
        for i in range(len(outputs)):
            outputs[i] = optimizations.send(adjust(outputs, inputs[i]))


def adjust(outputs: Iterable[Tuple[ndarray, ndarray]], ideal: ndarray) -> ndarray:
    return sum(contract((ideal - out[0]), out[1], axes=0) for out in outputs)


def cost(labels1: ndarray, labels2: ndarray) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2
