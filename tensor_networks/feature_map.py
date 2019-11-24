from math import pi, sin, cos

import numpy as np

from tensor_networks.utils.annotations import *


def color_abs_to_percentage(value: AbsColor) -> PartialColor:
    return value / 255


def feature(percentage: PartialColor) -> ndarray:
    """
    :return: [black value, white value] with black value + white value == 1
    """
    return np.array([cos(pi / 2 * percentage), sin(pi / 2 * percentage)])


def map_input(values: ndarray) -> Iterator[ndarray]:
    return map(feature, map(color_abs_to_percentage, values))
