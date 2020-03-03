# flake8: noqa
import logging

import numpy as np
from scipy.sparse.linalg import svds as scipy_sparse_svd

from tensor_networks.annotations import *


def sparse_svd(a: Array) -> Tuple[Array, Array, Array]:
    new_a = np.zeros([np.shape(a)[0] + 2, np.shape(a)[1] + 2])
    new_a[0:np.shape(a)[0], 0:np.shape(a)[1]] = a
    u, s, v = scipy_sparse_svd(new_a, k=min(new_a.shape[0], new_a.shape[1]) - 2)
    u = np.fliplr(u[:-2, :])
    s = s[::-1].real
    v = np.flipud(v[:, :-2])
    return u, s, v


def robust_svd(a: Array, tolerance: float = 0.0 * 1e-14) -> Tuple[Array, Array, Array]:
    try:
        u, s, v = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError as e1:
        logging.warning(f'Ran into {type(e1).__name__}: {e1}... transposing and retrying...')
        try:
            # TODO: When is this supposed to help?
            v, s, u = np.linalg.svd(np.transpose(a), full_matrices=False)
            u = np.transpose(u)
            v = np.transpose(v)
            logging.debug('Transposing succeeded!')
        except np.linalg.LinAlgError as e2:
            logging.warning(f'Ran into {type(e2).__name__}: {e2}... calling sparse code...')
            u, s, v = sparse_svd(a)
            logging.debug('Sparse code succeeded!')
    # TODO: The next lines break some decompositions. Investigate their purpose
    u = u[:, ~np.isclose(s, 0.0, atol=tolerance)]
    s = s[~np.isclose(s, 0.0, atol=tolerance)]
    v = v[np.transpose(~np.isclose(s, 0.0, atol=tolerance)), :]
    return u, s, v
