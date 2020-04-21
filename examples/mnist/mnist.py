"""
TODO: explain this example
"""
from functools import partial

from examples.utils.greyscale_image import image_feature
from examples.utils.io import load_mat_data_set, print_guesses
from tensor_networks.decomposition import truncated_svd
from tensor_networks.inputs import index_label
from tensor_networks.patched_numpy import np
from tensor_networks.training import sweep_entire_train
from tensor_networks.weights import starting_weights


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
    print()
    sweep_iterator = sweep_entire_train(weights, train_inputs,
                                        svd=partial(truncated_svd, max_chi=20))
    for i in range(1, 11):
        print(f'### Sweep {i} ###')
        next(sweep_iterator)
        print_guesses(test_inputs, weights)
        print()
