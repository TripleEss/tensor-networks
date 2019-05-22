import numpy as np

if __name__ == '__main__':
    A = np.arange(9).reshape(3, 3)
    B = np.arange(27).reshape(3, 3, 3)
    C = np.arange(9).reshape(3, 3)
    D = np.arange(27).reshape(3, 3, 3)
    network = np.tensordot(A, B, [(1,), (2,)])
    network = np.tensordot(network, D, [(0, 2), (0, 1)])
    network = np.tensordot(network, C, [(0, 1), (1, 0)])
