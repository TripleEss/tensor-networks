from functools import partial
from itertools import tee, chain, islice, cycle

import numpy as np
from more_itertools import consume

from tensor_networks.annotations import *
from tensor_networks.contraction import contract
from tensor_networks.inputs import Input
from tensor_networks.svd import truncated_svd, split
from tensor_networks.tensor_train import TensorTrain


_T = TypeVar('_T')


def tee_zip_async(seq: Sequence[_T], start: int = 0, direction: int = 1
                  ) -> Iterator[Tuple[_T, _T]]:
    seq_rev = list(reversed(seq))[1:-1]
    iter1, iter2 = tee(cycle(chain(seq, seq_rev)))
    next(iter2, None)
    if direction == -1:
        start = 2 * (len(seq) - 1) - start
    return islice(zip(iter1, iter2), start, None)


def update(ideals: Iterable[Array], outputs: Iterable[Array],
           inputs: Iterable[Array]) -> Array:
    # TODO only use a small multiple of result
    result = - 0.000000001 * sum(contract((idl - out), inp, axes=0).transpose(1, 2, 0, 3, 4)
                                 for idl, out, inp in zip(ideals, outputs, inputs))
    assert isinstance(result, Array)
    return result


def sweep(ttrain: TensorTrain, inputs: Sequence[Input], label_index: int = 0,
          direction: int = 1, updater: Updater = None,
          svd: SVDCallable = truncated_svd) -> Generator[None, None, None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param ttrain: The train to optimize
    :param inputs: TODO
    :param label_index: The index at which we start optimizing
    :param direction: The direction in which we start sweeping
    :param updater: The function used for calculating updates
    :param svd: The function used for singular value decomposition
    """
    assert len(inputs[0]) == len(ttrain)

    if updater is None:
        updater = partial(update, [inp.label for inp in inputs])

    left_i, right_i = sorted((label_index, label_index + direction))
    acc_lefts = [list(ttrain[:left_i]
                      .attach(inp[:left_i])
                      .contractions(keep_mock_index=False))
                 for inp in inputs]
    acc_rights = [list(ttrain[:right_i:-1]
                       .attach(inp[:right_i:-1])
                       .contractions(keep_mock_index=False))
                  for inp in inputs]

    for label_index, other_index in tee_zip_async(range(len(ttrain)),
                                                  start=label_index,
                                                  direction=direction):
        yield
        direction = 1 if label_index < other_index else -1
        label_core, other_core = ttrain[label_index], ttrain[other_index]

        # core with label is contracted with the next core
        to_optimize = contract(label_core, other_core)

        outputs = []
        local_inputs: List[Array] = []
        for inp, acc_left, acc_right in zip(inputs, acc_lefts, acc_rights):
            label_input, other_input = inp[label_index], inp[other_index]
            left_reduced = acc_left[-1] if acc_left else np.array([1])
            right_reduced = acc_right[-1] if acc_right else np.array([1])
            label_reduced, other_reduced = (left_reduced, right_reduced)[::direction]
            # tensor product of all inputs
            local_inp = contract(label_reduced, label_input, other_input,
                                 other_reduced, axes=0)
            local_inputs.append(local_inp)
            outputs.append(contract(to_optimize, local_inp,
                                    axes=([0, 1, 3, 4], [0, 1, 2, 3])))

        optimized = to_optimize + updater(outputs, local_inputs)
        optimized *= np.linalg.norm(to_optimize) / np.linalg.norm(optimized)
        label_core, other_core = split(optimized, before_index=2, svd=svd)
        # transpose label index
        other_core = other_core.transpose(0, 2, 1, 3)
        ttrain[label_index], ttrain[other_index] = label_core, other_core

        for inp, acc_left, acc_right in zip(inputs, acc_lefts, acc_rights):
            label_acc, other_acc = (acc_left, acc_right)[::direction]
            label_input = inp[label_index]
            if label_acc:
                reduced_next = contract(*(label_acc[-1], label_core)[::direction])
            else:
                reduced_next = label_core.reshape(label_core.shape[1:])
            label_acc.append(contract(*(label_input, reduced_next)[::direction]))
            other_acc.pop()


def sweep_until(tensor_train: TensorTrain, inputs: Sequence[Input],
                iterations: Optional[int] = None, **kwargs) -> None:
    # TODO: other break conditions
    isweep = sweep(tensor_train, inputs, **kwargs)
    consume(isweep, iterations)
