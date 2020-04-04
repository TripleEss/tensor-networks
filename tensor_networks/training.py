from functools import partial
from itertools import tee

import numpy as np
from more_itertools import consume

from tensor_networks import utils
from tensor_networks.utils import Direction
from tensor_networks.annotations import *
from tensor_networks.contraction import contract, tensor_product, attach
from tensor_networks.inputs import Input
from tensor_networks.svd import truncated_svd, split
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.transposition import transpose_bond_indices


def update(ideals: Iterable[Array], outputs: Iterable[Array],
           inputs: Iterable[Array]) -> Array:
    full_update = sum(contract((ideal - out), inp, axes=0).transpose(1, 2, 0, 3, 4)
                      for ideal, out, inp in zip(ideals, outputs, inputs))
    # TODO: make factor a variable
    small_update = - 0.000000001 * full_update
    assert isinstance(small_update, Array)
    return small_update


def output(to_optimize: Array, local_in: Array) -> Array:
    return contract(to_optimize, local_in, axes=([0, 1, 3, 4], [0, 1, 2, 3]))


def outputs(to_optimize: Array, local_inputs: Iterable[Array]
            ) -> Iterator[Array]:
    return map(partial(output, to_optimize), local_inputs)


def shift_labels(optimized_label_core: Array,
                 label_inputs: Iterable[Array],
                 label_accs: Iterable[List[Array]],
                 other_accs: Iterable[List[Array]]) -> None:
    for li, la, oa in zip(label_inputs, label_accs, other_accs):
        shift_label(optimized_label_core,
                    label_input=li, label_acc=la, other_acc=oa)


def shift_label(optimized_label_core: Array, label_input: Array,
                label_acc: List[Array], other_acc: List[Array]) -> None:
    """
    modifies label_acc and other_acc
    """
    # if there are no cores before optimized_label_core then
    # we simply remove the mock index (by contracting with ONE_TENSOR)
    previous_label_reduced = utils.get_last_or_one_tensor(label_acc)
    new_label_reduced = contract(
        previous_label_reduced,
        attach(optimized_label_core, label_input)
    )
    label_acc.append(new_label_reduced)

    # pop the last element off of other_acc
    if other_acc:
        del other_acc[-1]


def sweep(ttrain: TensorTrain,
          inputs: Sequence[Input],
          label_index: int = 0,
          starting_direction: Direction = Direction.LEFT_TO_RIGHT,
          updater: Updater = None,
          svd: SVDCallable = truncated_svd
          ) -> Generator[None, None, None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param ttrain: The train to optimize
    :param inputs: TODO
    :param label_index: The index at which we start optimizing
    :param starting_direction: The direction in which we start sweeping
    :param updater: The function used for calculating updates
    :param svd: The function used for singular value decomposition
    """
    assert len(inputs[0]) == len(ttrain)

    if updater is None:
        updater = partial(update, [inp.label for inp in inputs])

    index_generator1, index_generator2 = tee(
        utils.swing_pairwise(range(len(ttrain)),
                             start=label_index,
                             direction=starting_direction)
    )

    label_index, other_index, direction = next(index_generator1)
    left_index, right_index = ((label_index, other_index)
                               if direction == Direction.LEFT_TO_RIGHT
                               else (other_index, label_index))

    # Initialize accumulated inputs from the left and the right.
    # Using accumulation instead of reduction
    # (which would only be the last element of the accumulation)
    # has the advantage of avoiding redundant computation since only the ends
    # of the accumulation ever change and the rest of it can be reused.
    acc_lefts = [list(ttrain[:left_index]
                      .attach(inp[:left_index])
                      .contractions(keep_mock_index=False))
                 for inp in inputs]
    acc_rights = [list(ttrain[:right_index:-1]
                       .attach(inp[:right_index:-1])
                       .contractions(keep_mock_index=False))
                  for inp in inputs]

    for label_index, other_index, direction in index_generator2:
        # yield to allow the caller of this function to stop
        # the iteration at some point (otherwise this would go on infinitely)
        yield
        # swap bond indices when going backward so that further algorithms
        # only need to handle the case of going left to right
        maybe_transpose_bond_indices = (utils.identity
                                        if direction == Direction.LEFT_TO_RIGHT
                                        else transpose_bond_indices)
        # The prefixes 'l' and 'r' stand for 'left' and 'right'.
        # They symbolize that the variable can be used as if
        # it really was on the left/right side
        # (even though it might actually have been on the other side).
        l_label_accs, r_other_accs = ((acc_lefts, acc_rights)
                                      if direction == Direction.LEFT_TO_RIGHT
                                      else (acc_rights, acc_lefts))
        l_label_reduceds = map(utils.get_last_or_one_tensor, l_label_accs)
        r_other_reduceds = map(utils.get_last_or_one_tensor, r_other_accs)

        label_inputs = [inp[label_index] for inp in inputs]
        other_inputs = [inp[other_index] for inp in inputs]

        local_inputs = map(tensor_product,
                           l_label_reduceds, label_inputs,
                           other_inputs, r_other_reduceds)

        l_label_core = maybe_transpose_bond_indices(ttrain[label_index])
        r_other_core = maybe_transpose_bond_indices(ttrain[other_index])
        # core with label is contracted with the next core
        to_optimize = contract(l_label_core, r_other_core)
        outs = outputs(to_optimize=to_optimize, local_inputs=local_inputs)

        optimized = to_optimize + updater(outs, local_inputs)
        optimized *= np.linalg.norm(to_optimize) / np.linalg.norm(optimized)
        l_label_core, r_other_core = split(optimized, before_index=2, svd=svd)
        # transpose label index into its correct position (from 1 to 2)
        r_other_core = r_other_core.swapaxes(1, 2)
        ttrain[label_index] = maybe_transpose_bond_indices(l_label_core)
        ttrain[other_index] = maybe_transpose_bond_indices(r_other_core)

        shift_labels(optimized_label_core=l_label_core,
                     label_inputs=label_inputs,
                     label_accs=l_label_accs,
                     other_accs=r_other_accs)


def sweep_until(tensor_train: TensorTrain, inputs: Sequence[Input],
                iterations: Optional[int] = None, **kwargs) -> None:
    # TODO: more break conditions
    isweep = sweep(tensor_train, inputs, **kwargs)
    consume(isweep, iterations)
