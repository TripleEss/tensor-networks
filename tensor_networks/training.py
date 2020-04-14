from functools import partial
from itertools import tee

from more_itertools import consume

from tensor_networks.patched_numpy import np
from tensor_networks import utils
from tensor_networks.utils import Direction, neutral_array
from tensor_networks.annotations import *
from tensor_networks.contraction import contract, tensor_product, attach
from tensor_networks.inputs import Input
from tensor_networks.decomposition import truncated_svd, split
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.transposition import transpose_outer_indices


def update(ideal: Array, output: Array, input_: Array) -> Array:
    """
    Calculate an update for two contracted cores based on an input.
    The returned update array has the same shape as the two contracted cores
    and values that are relatively small compared to the cores.
    """
    full_update = tensor_product(ideal - output, input_).transpose(1, 2, 0, 3, 4)
    # TODO: make factor a variable
    small_update = 0.001 * full_update
    assert isinstance(small_update, Array)
    return small_update


def calculate_output(to_optimize: Array, local_in: Array) -> Array:
    return contract(to_optimize, local_in, axes=([0, 1, 3, 4], [0, 1, 2, 3]))


def shift_accumulations(optimized_label_core: Array, label_input: Array,
                        label_acc: List[Array], other_acc: List[Array]) -> None:
    """
    modifies label_acc and other_acc
    """
    # If we've reached the end of the tensor train then we
    # 1. can't pop the last element off of other_acc since it is empty and
    # 2. therefore also don't want to append to label_acc since otherwise
    #    the amount of elements in label_acc and other_acc combined would
    #    increase.
    if not other_acc:
        return

    # if there are no cores before optimized_label_core then
    # we simply remove its mock index (by contracting with ONE_TENSOR)
    previous_label_reduced = utils.get_last(label_acc, default=neutral_array(1))
    new_label_reduced = contract(
        previous_label_reduced,
        attach(optimized_label_core, label_input)
    )
    label_acc.append(new_label_reduced)

    # pop the last element off of other_acc
    del other_acc[-1]


def sweep(ttrain: TensorTrain,
          inputs: Sequence[Input],
          label_index: int = 0,
          starting_direction: Direction = Direction.LEFT_TO_RIGHT,
          updater: Updater = update,
          svd: SVDCallable = truncated_svd
          ) -> Iterator[None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param ttrain: The train to optimize
    :param inputs: TODO
    :param label_index: The index at which we start optimizing
    :param starting_direction: The direction in which we start sweeping
    :param updater: The function used for calculating updates
    :param svd: The function used for singular value decomposition
    """
    assert len(inputs[0].array) == len(ttrain)

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
                      .attach(inp.array[:left_index])
                      .contractions(keep_mock_index=False))
                 for inp in inputs]
    acc_rights = [list(ttrain[:right_index:-1]
                       .attach(inp.array[:right_index:-1])
                       .contractions(keep_mock_index=False))
                  for inp in inputs]

    for label_index, other_index, direction in index_generator2:
        # yield to allow the caller of this function to stop
        # the iterations at some point (otherwise this would go on infinitely)
        yield

        # The prefixes 'l' and 'r' stand for 'left' and 'right'.
        # They symbolize that the variable can be used as if
        # it really was on the left/right side
        # (even though it might actually have been on the other side).
        l_label_accs, r_other_accs = ((acc_lefts, acc_rights)
                                      if direction == Direction.LEFT_TO_RIGHT
                                      else (acc_rights, acc_lefts))
        l_label_reduceds = [utils.get_last(acc, default=neutral_array(1))
                            for acc in l_label_accs]
        r_other_reduceds = [utils.get_last(acc, default=neutral_array(1))
                            for acc in r_other_accs]

        label_inputs = [inp.array[label_index] for inp in inputs]
        other_inputs = [inp.array[other_index] for inp in inputs]

        local_inputs = [tensor_product(l_label_r, label_i, other_i, r_other_r)
                        for l_label_r, label_i, other_i, r_other_r
                        in zip(l_label_reduceds, label_inputs,
                               other_inputs, r_other_reduceds)]

        # swap bond indices when going backward so that further algorithms
        # only need to handle the case of going left to right
        maybe_transpose_bond_indices: Callable[[Array], Array] = (
            utils.identity  # type: ignore[assignment]
            if direction == Direction.LEFT_TO_RIGHT
            else transpose_outer_indices
        )
        l_label_core = maybe_transpose_bond_indices(ttrain[label_index])
        r_other_core = maybe_transpose_bond_indices(ttrain[other_index])
        # core with label is contracted with the next core
        to_optimize = contract(l_label_core, r_other_core)
        outputs = [calculate_output(to_optimize, local_in)
                   for local_in in local_inputs]

        update_ = sum(updater(label_vec, out, local_in)
                      for label_vec, out, local_in
                      in zip((inp.label for inp in inputs), outputs, local_inputs))
        optimized = to_optimize + update_
        optimized *= np.linalg.norm(to_optimize) / np.linalg.norm(optimized)
        l_label_core, r_other_core = split(optimized, before_index=2, svd=svd)
        # transpose label index into its correct position (from 1 to 2)
        r_other_core = r_other_core.swapaxes(1, 2)
        ttrain[label_index] = maybe_transpose_bond_indices(l_label_core)
        ttrain[other_index] = maybe_transpose_bond_indices(r_other_core)

        for li, la, oa in zip(label_inputs, l_label_accs, r_other_accs):
            shift_accumulations(l_label_core, label_input=li,
                                label_acc=la, other_acc=oa)


def sweep_until(tensor_train: TensorTrain, inputs: Sequence[Input],
                iterations: Optional[int] = None, **kwargs) -> None:
    # TODO: more break conditions
    isweep = sweep(tensor_train, inputs, **kwargs)
    consume(isweep, iterations)
