from tensor_networks.annotations import *

import numpy as np

from tensor_networks.inputs import Input
from tensor_networks.tensor_train import TensorTrain


def cost(labels1: Array, labels2: Array) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2


def classify(ttrain: TensorTrain, input: Input):
    return ttrain.attach(input).contract(fully=True)
