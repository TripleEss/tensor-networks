from tensor_networks.patched_numpy import np


def index_label(label: int, maximum_index: int):
    array = np.zeros(maximum_index + 1)
    array[label] = 1
    return array
