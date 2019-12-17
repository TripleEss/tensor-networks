import numpy as np
from scipy.io import loadmat

from tensor_networks.classification import cost, classify
from tensor_networks.feature import img_feature
from tensor_networks.patch_dtype import patch_dtype
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.training import sweep_until


# patch arrays to be float32 to enhance performance
patch_dtype(np.float32)

# load data set
data = loadmat('./mnist/MNIST.mat', squeeze_me=True)
train_amount = test_amount = 10
trainX, trainY = data['trainX'][:train_amount], data['trainY'][:train_amount]
testX, testY = data['testX'][:test_amount], data['testY'][:test_amount]
train_inputs = list(map(img_feature, trainX, trainY))
test_inputs = list(map(img_feature, testX, testY))

# starting weights
weights = TensorTrain([
    np.ones((1, 2, 10, 2)),
    *[np.ones((2, 2, 2))] * (len(train_inputs[0]) - 2),
    np.ones((2, 2, 1)),
])

# optimize
sweep_until(weights, train_inputs, iterations=5)
for test_inp in test_inputs:
    classified_label = classify(weights, test_inp)
    cost_ = cost(classified_label, test_inp.label)
    guess = np.max(classified_label)
    actual = np.max(test_inp.label)
    print(f'{guess=}, {actual=}, {cost_=}, {classified_label=}')
