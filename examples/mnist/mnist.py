"""
TODO: explain this example
"""
from functools import partial

from more_itertools import consume

from examples.utils.greyscale_image import image_feature
from examples.utils.io import load_mat_data_set, print_guesses
from tensor_networks.inputs import index_label
from tensor_networks.weights import starting_weights
from tensor_networks.patched_numpy import np
from tensor_networks.decomposition import truncated_svd
from tensor_networks.training import sweep


if __name__ == '__main__':
    # patch arrays to be float32 to enhance performance
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32

    # load data set
    train_inputs, test_inputs = load_mat_data_set(
        path='./examples/mnist/MNIST.mat',
        feature=lambda x, y: (image_feature(x), index_label(y, 9)),
        train_amount=50,
        test_amount=50
    )

    # starting weights
    weights = starting_weights(input_length=len(train_inputs[0].array),
                               label_length=10)

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
            print()
