from functools import partial

from tensor_networks.patched_numpy import np
from more_itertools import consume
from scipy.io import loadmat

from tensor_networks.classification import cost, classify
from tensor_networks.feature import img_feature
from tensor_networks.svd import truncated_svd
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.training import sweep


# patch arrays to be float32 to enhance performance
np.GLOBAL_NUMERIC_DATA_TYPE = np.float32

# load data set
data = loadmat('./mnist/MNIST.mat', squeeze_me=True)
train_amount = test_amount = 50
trainX, trainY = data['trainX'][:train_amount], data['trainY'][:train_amount]
testX, testY = data['testX'][:test_amount], data['testY'][:test_amount]
train_inputs = list(map(img_feature, trainX, trainY))
test_inputs = list(map(img_feature, testX, testY))

# starting weights
weights = TensorTrain([
    np.ones((1, 2, 10, 2)),
    *[np.array(2 * list(np.eye(2, 2))).reshape(2, 2, 2).transpose(2, 0, 1)] * (len(train_inputs[0].array) - 2),
    np.ones((2, 2, 1)),
]) / 1.0286


def print_guesses():
    for test_inp in test_inputs:
        classified_label = classify(weights, test_inp)
        cost_ = cost(classified_label, test_inp.label)
        guess = np.argmax(classified_label)
        actual = np.argmax(test_inp.label)
        print(f'{guess=}, {actual=}, cost={round(cost_, 2)}\n'
              f'\tlabel vector: {list(np.round(classified_label, 1))}')


# optimize
print_guesses()
sweeper = sweep(weights, train_inputs, svd=partial(truncated_svd, max_chi=20))
logging_interval = len(weights) * 4
for i in range(0, logging_interval * 3 + 1):
    if i % logging_interval == 0:
        print_guesses()
        print(f'### Iterations {i} - {i + logging_interval} ###')
    consume(sweeper, 1)
