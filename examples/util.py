from scipy.io import loadmat

from tensor_networks.classification import classify, cost
from tensor_networks.patched_numpy import np


def load_mat_data_set(path, feature, train_amount=None, test_amount=None):
    data = loadmat(path, squeeze_me=True)

    trainX, trainY = data['trainX'][:train_amount], data['trainY'][:train_amount]
    testX, testY = data['testX'][:test_amount], data['testY'][:test_amount]

    train_inputs = list(map(feature, trainX, trainY))
    test_inputs = list(map(feature, testX, testY))
    return train_inputs, test_inputs


def print_guesses(test_inputs, weights, decimal_places=2):
    cost_sum = 0
    for test_inp in test_inputs:
        classified_label = classify(weights, test_inp)
        cost_ = cost(classified_label, test_inp.label)
        cost_sum += cost_
        guess = np.argmax(classified_label)
        actual = np.argmax(test_inp.label)
        print(f'{guess=}, {actual=}, cost={round(cost_, decimal_places)}\n'
              f'\tlabel vector: '
              f'{list(np.round(classified_label, decimal_places))}')

    print(f'Overall cost: {cost_sum}')
