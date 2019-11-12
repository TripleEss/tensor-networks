import logging

import numpy as np

from utils.setup_logging import setup_logging
from tensor_networks.tensor_train import TensorTrain


if __name__ == '__main__':
    setup_logging()
    logging.debug('--- Testing tensor decomposition ---')
    t = np.arange(64)
    logging.debug(f"Original tensor's shape: {t.shape}")
    ttrain = TensorTrain.decompose(t, d=2, chi=2)
    logging.debug(f"Shape of decomposed tensor train: {[x.shape for x in ttrain]}")
    ttest = ttrain.reduce()
    logging.debug(f"Shape of the reassembled tensor train: {ttest.shape}")
    diff = abs(t.flatten() - ttest.flatten()).reshape(ttest.shape)
    logging.debug(f"Maximum difference between the original and the reassembled tensor: {diff.max()}")