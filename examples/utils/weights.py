from tensor_networks.patched_numpy import np
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.utils import neutral_array


def starting_weights(input_length: int, label_length: int) -> TensorTrain:
    return TensorTrain(
        [np.stack([neutral_array(1, 2, 2)] * label_length, axis=2)]
        + [neutral_array(2, 2, 2) for _ in range(input_length - 2)]
        + [neutral_array(2, 2, 1)]
    )
