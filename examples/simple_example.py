import random
from functools import partial
from itertools import islice

from more_itertools import consume
from scipy.io import savemat

from examples import util
from tensor_networks.annotations import *
from tensor_networks.feature import Input, color_abs_to_percentage, feature
from tensor_networks.patched_numpy import np

# from left to right:
# 0 == bright to dark
# 1 == dark to bright
from tensor_networks.svd import truncated_svd
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.training import sweep


FILE_PATH = './data/2x2-gradient/2x2.mat'


def generate_data_set():
    while True:
        b0 = 0  # random.random() * 0.6
        b1 = 0  # random.random() * 0.6
        d0 = 255  # min(b0 + 0.5 * random.random(), 1.0)
        d1 = 255  # min(b1 + 0.5 * random.random(), 1.0)
        label = random.choice([0, 1])
        if label == 0:
            yield np.array([b0, d0, b1, d1]), 0
        else:
            yield np.array([d0, b0, d1, b1]), 1


def save_data_set(n):
    trainX, trainY = zip(*list(islice(generate_data_set(), n)))
    testX, testY = zip(*list(islice(generate_data_set(), n)))
    trainX = np.array(list(trainX))
    trainY = np.array(list(trainY))
    testX = np.array(list(testX))
    testY = np.array(list(testY))
    savemat(FILE_PATH, {
        'trainX': trainX,
        'trainY': trainY,
        'testX': testX,
        'testY': testY,
    })


def img_feature(values: Array, label: int) -> Input:
    return Input(
        np.array(list(map(feature, map(color_abs_to_percentage, values)))),
        label=np.array([1, 0]) if label == 0 else np.array([0, 1]),
    )


if __name__ == '__main__':
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32
    train_inputs, test_inputs = util.load_mat_data_set(
        path=FILE_PATH,
        feature=img_feature,
        train_amount=500,
        test_amount=50,
    )

    # starting weights
    weights = TensorTrain([
        np.ones((1, 2, 2, 2)),
        *([np.array(2 * list(np.eye(2, 2)))
          .reshape(2, 2, 2)
          .transpose(2, 0, 1)] * (len(train_inputs[0].array) - 2)),
        np.ones((2, 2, 1)),
    ]) / 1.0286

    # optimize
    util.print_guesses(test_inputs, weights, decimal_places=10)
    sweeper = sweep(weights, train_inputs, svd=partial(truncated_svd, max_chi=20))
    logging_interval = len(weights)
    for i in range(1, logging_interval * 4):
        if i % logging_interval == 0:
            print(f'### Iterations {i} - {i + logging_interval -1} ###')
        consume(sweeper, 1)
        if i % logging_interval == logging_interval - 1:
            util.print_guesses(test_inputs, weights, decimal_places=10)
