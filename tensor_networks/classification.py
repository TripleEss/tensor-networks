from tensor_networks.annotations import *

import numpy as np

from tensor_networks.inputs import Input


def cost(labels1: ndarray, labels2: ndarray) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2


def classify(ttrain: TTrain, input: Input):
    return ttrain.attach(input).contract(fully=True)
