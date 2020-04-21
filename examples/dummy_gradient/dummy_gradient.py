"""
Very basic example used for debugging.
The input consists of 2x2 greyscale images where the left pixels are either
brighter (label 1) than the pixels on the right or darker (label 0).
"""
from functools import partial

from examples.utils.greyscale_image import image_feature
from examples.utils.io import load_mat_data_set, print_guesses
from tensor_networks.decomposition import truncated_svd
from tensor_networks.inputs import index_label
from tensor_networks.patched_numpy import np
from tensor_networks.training import sweep_entire_train
from tensor_networks.weights import starting_weights


FILE_PATH = './examples/dummy_gradient/2x2-gradients.mat'


if __name__ == '__main__':
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32
    train_inputs, test_inputs = load_mat_data_set(
        path=FILE_PATH,
        feature=lambda x, y: (image_feature(x), index_label(y, 1)),
        train_amount=500,
        test_amount=50,
    )

    # starting weights
    weights = starting_weights(input_length=len(train_inputs[0].array),
                               label_length=2)

    # optimize
    print_guesses(test_inputs, weights)
    print()
    sweep_iterator = sweep_entire_train(weights, train_inputs,
                                        svd=partial(truncated_svd, max_chi=20))
    for i in range(1, 4):
        print(f'### Sweep {i} ###')
        next(sweep_iterator)
        print_guesses(test_inputs, weights)
        print()
