from functools import partial

from more_itertools import consume

from examples import util
from tensor_networks.feature import img_feature
from tensor_networks.patched_numpy import np
from tensor_networks.svd import truncated_svd
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.training import sweep


if __name__ == '__main__':
    # patch arrays to be float32 to enhance performance
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32

    # load data set
    train_inputs, test_inputs = util.load_mat_data_set(
        path='./data/mnist/MNIST.mat',
        feature=img_feature,
        train_amount=50,
        test_amount=50
    )

    # starting weights
    weights = TensorTrain([
        np.ones((1, 2, 10, 2)),
        * ([np.array(2 * list(np.eye(2, 2)))
            .reshape(2, 2, 2)
            .transpose(2, 0, 1)] * (len(train_inputs[0].array) - 2)),
        np.ones((2, 2, 1)),
    ]) / 1.0286

    # optimize
    util.print_guesses(test_inputs, weights)
    sweeper = sweep(weights, train_inputs, svd=partial(truncated_svd, max_chi=20))
    logging_interval = len(weights) * 4
    for i in range(0, logging_interval * 3 + 1):
        if i % logging_interval == 0:
            util.print_guesses(test_inputs, weights)
            print(f'### Iterations {i} - {i + logging_interval} ###')
        consume(sweeper, 1)
