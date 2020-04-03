from enum import Enum
from functools import partial
from itertools import tee, chain, islice, cycle

import numpy as np
from more_itertools import consume

from tensor_networks import utils
from tensor_networks.annotations import *
from tensor_networks.contraction import contract
from tensor_networks.inputs import Input
from tensor_networks.svd import truncated_svd, split
from tensor_networks.tensor_train import TensorTrain


class Direction(Enum):
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = -1


_T = TypeVar('_T')


def swing_pairwise(seq: Sequence[_T],
                   start: int = 0,
                   direction: Direction = Direction.LEFT_TO_RIGHT
                   ) -> Iterator[Tuple[_T, _T]]:
    seq_rev = list(reversed(seq))[1:-1]
    iter1, iter2 = tee(cycle(chain(seq, seq_rev)))
    next(iter2, None)
    if direction == Direction.RIGHT_TO_LEFT:
        start = 2 * (len(seq) - 1) - start
    return islice(zip(iter1, iter2), start, None)


def update(ideals: Iterable[Array], outputs: Iterable[Array],
           inputs: Iterable[Array]) -> Array:
    full_update = sum(contract((idl - out), inp, axes=0).transpose(1, 2, 0, 3, 4)
                      for idl, out, inp in zip(ideals, outputs, inputs))
    # TODO make factor a variable
    small_update = - 0.000000001 * full_update
    assert isinstance(small_update, Array)
    return small_update


def sweep_foo(to_optimize: Array,
              label_input: Array,
              other_input: Array,
              label_acc: List[Array],
              other_acc: List[Array]
              ) -> Tuple[Array, Array]:
    """
    :return: tuple of the local_input and the output
    """
    # label_acc, other_acc = (acc_left, acc_right)[::direction]
    label_reduced = utils.get_last(label_acc, default=utils.ONE_TENSOR)
    other_reduced = utils.get_last(other_acc, default=utils.ONE_TENSOR)

    # tensor product of all inputs
    local_input = contract(label_reduced, label_input,
                           other_input, other_reduced,
                           axes=0)

    output = contract(to_optimize, local_input,
                      axes=([0, 1, 3, 4], [0, 1, 2, 3]))

    return local_input, output


def sweep_bar_helper(optimized_label_core: Array,
                     direction: Direction,
                     label_input: Array,
                     label_acc: List[Array]
                     ) -> Array:
    if label_acc:
        if direction == Direction.LEFT_TO_RIGHT:
            label_reduced_core = contract(label_acc[-1], optimized_label_core)
        else:
            label_reduced_core = contract(optimized_label_core, label_acc[-1])
    else:
        # there are no cores before optimized_label_core
        # so we just remove the mock index
        mock_index = 0 if direction == Direction.LEFT_TO_RIGHT else -1
        label_reduced_core = optimized_label_core.squeeze(axis=mock_index)

    if direction == Direction.LEFT_TO_RIGHT:
        return contract(label_input, label_reduced_core)
    else:
        return contract(label_reduced_core, label_input)


def sweep_bar(optimized_label_core: Array,
              direction: Direction,
              label_input: Array,
              label_acc: List[Array],
              other_acc: List[Array]
              ) -> None:
    """
    modifies label_acc and other_acc
    """
    label_reduced = sweep_bar_helper(optimized_label_core,
                                     direction,
                                     label_input,
                                     label_acc)
    label_acc.append(label_reduced)

    if other_acc:
        del other_acc[-1]


def sweep_helper_pre(to_optimize: Array,
                     direction: Direction,
                     label_inputs: Iterable[Array],
                     other_inputs: Iterable[Array],
                     acc_lefts: Iterable[List[Array]],
                     acc_rights: Iterable[List[Array]]
                     ) -> Tuple[Iterable[Array], Iterable[Array]]:
    local_inputs: List[Array] = []
    outputs: List[Array] = []

    for label_input, other_input, acc_left, acc_right in \
            zip(label_inputs, other_inputs, acc_lefts, acc_rights):
        if direction == Direction.LEFT_TO_RIGHT:
            label_acc = acc_left
            other_acc = acc_right
        else:
            label_acc = acc_right
            other_acc = acc_left

        local_input, output = sweep_foo(to_optimize=to_optimize,
                                        label_input=label_input,
                                        other_input=other_input,
                                        label_acc=label_acc,
                                        other_acc=other_acc)

        local_inputs.append(local_input)
        outputs.append(output)

    return local_inputs, outputs


def sweep_helper_post(optimized_label_core: Array,
                      direction: Direction,
                      label_inputs: Iterable[Array],
                      acc_lefts: Iterable[List[Array]],
                      acc_rights: Iterable[List[Array]]):
    for label_input, acc_left, acc_right in zip(label_inputs, acc_lefts, acc_rights):
        if direction == Direction.LEFT_TO_RIGHT:
            label_acc = acc_left
            other_acc = acc_right
        else:
            label_acc = acc_right
            other_acc = acc_left

        sweep_bar(optimized_label_core=optimized_label_core,
                  direction=direction,
                  label_input=label_input,
                  label_acc=label_acc,
                  other_acc=other_acc)

        # if label_acc:
        #     reduced_next = contract(*(label_acc[-1], label_core)[::direction])
        # else:
        #     reduced_next = label_core.reshape(label_core.shape[1:])
        # label_acc.append(contract(*(label_input, reduced_next)[::direction]))
        # other_acc.pop()


def sweep(ttrain: TensorTrain,
          inputs: Sequence[Input],
          label_index: int = 0,
          direction: Direction = Direction.LEFT_TO_RIGHT,
          updater: Updater = None,
          svd: SVDCallable = truncated_svd
          ) -> Generator[None, None, None]:
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

    left_i, right_i = sorted((label_index, label_index + direction.value))
    acc_lefts = [list(ttrain[:left_i]
                      .attach(inp[:left_i])
                      .contractions(keep_mock_index=False))
                 for inp in inputs]
    acc_rights = [list(ttrain[:right_i:-1]
                       .attach(inp[:right_i:-1])
                       .contractions(keep_mock_index=False))
                  for inp in inputs]

    for label_index, other_index in swing_pairwise(range(len(ttrain)),
                                                   start=label_index,
                                                   direction=direction):
        yield
        direction = Direction.LEFT_TO_RIGHT \
            if label_index < other_index \
            else Direction.RIGHT_TO_LEFT
        label_core = ttrain[label_index]
        other_core = ttrain[other_index]

        # core with label is contracted with the next core
        to_optimize = contract(label_core, other_core)

        label_inputs = [inp[label_index] for inp in inputs]
        other_inputs = [inp[other_index] for inp in inputs]

        local_inputs, outputs = sweep_helper_pre(to_optimize=to_optimize,
                                                 direction=direction,
                                                 label_inputs=label_inputs,
                                                 other_inputs=other_inputs,
                                                 acc_lefts=acc_lefts,
                                                 acc_rights=acc_rights)

        optimized = to_optimize + updater(outputs, local_inputs)
        optimized *= np.linalg.norm(to_optimize) / np.linalg.norm(optimized)
        label_core, other_core = split(optimized, before_index=2, svd=svd)
        # transpose label index
        other_core = other_core.swapaxes(1, 2)
        ttrain[label_index] = label_core
        ttrain[other_index] = other_core

        sweep_helper_post(optimized_label_core=label_core,
                          direction=direction,
                          label_inputs=label_inputs,
                          acc_lefts=acc_lefts,
                          acc_rights=acc_rights)


def sweep_until(tensor_train: TensorTrain, inputs: Sequence[Input],
                iterations: Optional[int] = None, **kwargs) -> None:
    # TODO: other break conditions
    isweep = sweep(tensor_train, inputs, **kwargs)
    consume(isweep, iterations)
