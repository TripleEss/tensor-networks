import numpy as np
from scipy.io import loadmat

from tensor_networks.feature import img_feature
from tensor_networks.tensor_train import TensorTrain

from tensor_networks.training import sweep_simultaneously_until


# load data set
data = loadmat('./mnist/MNIST.mat', squeeze_me=True)
train_amount = test_amount = 10
trainX, trainY = data['trainX'][:train_amount], data['trainY'][:train_amount]
testX, testY = data['testX'][:test_amount], data['testY'][:test_amount]
train_inputs = np.array(list(map(img_feature, trainX, trainY)))
test_inputs = np.array(list(map(img_feature, testX, testY)))

# starting weights
weights = TensorTrain([
    np.ones((1, 2, 2)),
    *[np.ones((2, 2, 2))] * (len(train_inputs[0]) - 2),
    np.ones((2, 2, 1)),
])

# optimize
sweep_simultaneously_until(weights, train_inputs)

# TODO: test result
