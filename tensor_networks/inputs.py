from tensor_networks.annotations import *


class Input(NamedTuple):
    """Represents a featured input and its label array"""
    array: Array
    label: Array
