from math import pi, sin, cos

from tensor_networks.patched_numpy import np

from tensor_networks.annotations import *


def color_abs_to_percentage(value: AbsColor) -> PartialColor:
    return value / 255


def feature(percentage: PartialColor) -> Array:
    """
    :return: [black value, white value] with black value + white value == 1
    """
    return np.array([cos(pi / 2 * percentage), sin(pi / 2 * percentage)])


def label_to_vec(label: int) -> Array:
    return np.array(label * [0] + [1] + (9 - label) * [0])


class Input(NamedTuple):
    array: Array
    label: Array


def img_feature(values: Array, label: int) -> Input:
    return Input(
        np.array(list(map(feature, map(color_abs_to_percentage, values)))),
        label_to_vec(label),
    )
