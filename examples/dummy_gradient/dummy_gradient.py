"""
TODO: explain this example
from left to right:
0 == bright to dark
1 == dark to bright
"""
import random
from functools import partial
from itertools import islice

from more_itertools import consume
from scipy.io import savemat

from examples.utils.greyscale_image import image_feature_with_index_label
from examples.utils.io import load_mat_data_set, print_guesses
from examples.utils.weights import starting_weights
from tensor_networks.patched_numpy import np
from tensor_networks.svd import truncated_svd
from tensor_networks.training import sweep


FILE_PATH = './examples/dummy_gradient/2x2-gradients.mat'


def generate_data_set():
    while True:
        b0 = random.randint(0, 140)
        b1 = random.randint(0, 140)
        d0 = min(b0 + random.randint(0, 127), 255)
        d1 = min(b1 + random.randint(0, 127), 255)
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


if __name__ == '__main__':
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32
    train_inputs, test_inputs = load_mat_data_set(
        path=FILE_PATH,
        feature=image_feature_with_index_label(maximum_index=1),
        train_amount=500,
        test_amount=50,
    )

    # starting weights
    weights = starting_weights(input_length=len(train_inputs[0].array),
                               label_length=2)

    # optimize
    print_guesses(test_inputs, weights)
    sweeper = sweep(weights, train_inputs, svd=partial(truncated_svd, max_chi=20))
    logging_interval = len(weights)
    for i in range(1, logging_interval * 10):
        if i % logging_interval == 0:
            print(f'### Iterations {i} - {i + logging_interval - 1} ###')
        consume(sweeper, 1)
        if i % logging_interval == logging_interval - 1:
            print_guesses(test_inputs, weights)
