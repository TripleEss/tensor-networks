from tensor_networks.annotations import *

from tensor_networks.patched_numpy import np
from tensor_networks.feature import Input
from tensor_networks.tensor_train import TensorTrain


def cost(labels1: Array, labels2: Array) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2


def classify(ttrain: TensorTrain, input: Input):
    return ttrain.attach(input.array).contract(fully=True)
