from math import pi, sin, cos

import numpy as np

from tensor_networks.annotations import *
from tensor_networks.inputs import Input


def color_abs_to_percentage(value: AbsColor) -> PartialColor:
    return value / 255


def feature(percentage: PartialColor) -> Array:
    """
    :return: [black value, white value] with black value + white value == 1
    """
    return np.array([cos(pi / 2 * percentage), sin(pi / 2 * percentage)])


def label_to_vec(label: int):
    return np.array(label * [0] + [1] + (9 - label) * [0])


def img_feature(values: Array, label) -> Input:
    return Input(list(map(feature, map(color_abs_to_percentage, values))),
                 label=label_to_vec(label))
