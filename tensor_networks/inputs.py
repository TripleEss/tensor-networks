from dataclasses import dataclass

from tensor_networks.annotations import *
from tensor_networks.patched_numpy import np


@dataclass(frozen=True)
class Input:
    """Represents a featured input and its label array"""
    array: Array
    label: Array


def index_label(label: int, maximum_index: int):
    array = np.zeros(maximum_index + 1)
    array[label] = 1
    return array
