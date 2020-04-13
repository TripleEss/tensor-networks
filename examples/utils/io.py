from scipy.io import loadmat  # type: ignore[import]

from tensor_networks.classification import classify, cost
from tensor_networks.patched_numpy import np


def load_mat_data_set(path, feature, train_amount=None, test_amount=None):
    data = loadmat(path, squeeze_me=True)

    trainX, trainY = data['trainX'][:train_amount], data['trainY'][:train_amount]
    testX, testY = data['testX'][:test_amount], data['testY'][:test_amount]

    train_inputs = list(map(feature, trainX, trainY))
    test_inputs = list(map(feature, testX, testY))
    return train_inputs, test_inputs


def print_guesses(test_inputs, weights, decimal_places=None):
    cost_sum = 0
    correct_guesses = 0
    for test_inp in test_inputs:
        classified_label = classify(weights, test_inp)
        cost_ = cost(classified_label, test_inp.label)
        guess = np.argmax(classified_label)
        actual = np.argmax(test_inp.label)
        correct = guess == actual

        guess_vs_actual_str = (f'✅ {guess}'
                               if correct
                               else f'❌ {guess} ({actual=})')
        rounded_label = (classified_label
                         if decimal_places is None
                         else np.round(classified_label, decimal_places))
        print(f'{guess_vs_actual_str :15}; '
              f'cost={cost_} '
              f'{list(rounded_label)}')

        cost_sum += cost_
        if correct:
            correct_guesses += 1

    print(f'Overall cost: {cost_sum}\n'
          f'Success rate: {correct_guesses / len(test_inputs) :.0%} '
          f'({correct_guesses}/{len(test_inputs)})')
