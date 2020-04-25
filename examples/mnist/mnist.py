"""
TODO: explain this example
"""
import time
from functools import partial

from examples.utils.greyscale_image import image_feature, color_abs_to_percentage
from examples.utils.io import load_mat_data_set, print_test_results
from examples.utils.results import ManyClassificationTests
from tensor_networks.annotations import *
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
        path='./examples/mnist/mnist-14x14.mat',
        feature=lambda x, y: (image_feature(x), index_label(y, 9)),
        train_amount=1000,
        test_amount=1000
    )

    # starting weights
    weights = starting_weights(input_length=len(train_inputs[0].array),
                               label_length=10)

    # optimize
    all_the_tests: List[ManyClassificationTests] = []
    control_results = ManyClassificationTests.create(weights, test_inputs)
    all_the_tests.append(control_results)
    print_test_results(control_results, summary_only=True)
    print()
    sweep_iterator = sweep_entire_train(weights, train_inputs,
                                        svd=partial(truncated_svd, max_chi=20))
    for i in range(1, 21):
        print(f'### Sweep {i} ... ', end='')
        start_time = time.time()
        next(sweep_iterator)
        end_time = time.time()
        print(f'Done! ({end_time - start_time:.2f}s) ###')
        results = ManyClassificationTests.create(weights, test_inputs)
        all_the_tests.append(results)
        print_test_results(results, summary_only=True)
        print()
