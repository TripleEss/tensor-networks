import logging
from my_types import *

import numpy as np
from scipy.sparse.linalg import svds as scipy_sparse_svd


def sparse_svd(a: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    new_a = np.zeros([np.shape(a)[0] + 2, np.shape(a)[1] + 2], 'float64')
    new_a[0:np.shape(a)[0], 0:np.shape(a)[1]] = a
    u, s, v = scipy_sparse_svd(new_a, k=min(new_a.shape[0], new_a.shape[1]) - 2)
    u = np.fliplr(u[:-2, :])
    s = s[::-1].real
    v = np.flipud(v[:, :-2])
    return u, s, v


def robust_svd(a: ndarray, tolerance: float = 0.0 * 1e-14) -> Tuple[ndarray, ndarray, ndarray]:
    try:
        u, s, v = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError as e1:
        logging.warning(f'Ran into ({type(e1).__name__}: {e1})... transposing and retrying...')
        try:
            # How is this supposed to help?
            v, s, u = np.linalg.svd(np.transpose(a), full_matrices=False)
            u = np.transpose(u)
            v = np.transpose(v)
            logging.debug('Transposing succeeded!')
        except np.linalg.LinAlgError as e2:
            logging.warning(f'Ran into {type(e2).__name__}: {e2}... calling sparse code...')
            u, s, v = sparse_svd(a)
            logging.debug('Sparse code succeeded!')
    # The next lines break some decompositions. What's their purpose?
    u = u[:, ~np.isclose(s, 0.0, atol=tolerance)]
    s = s[~np.isclose(s, 0.0, atol=tolerance)]
    v = v[np.transpose(~np.isclose(s, 0.0, atol=tolerance)), :]
    return u, s, v
