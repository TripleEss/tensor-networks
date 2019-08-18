import logging
import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from typing import Tuple


USV = Tuple[np.ndarray, np.ndarray, np.ndarray]

def sparse_svd(a: np.ndarray) -> USV:
    new_a = np.zeros([np.shape(a)[0] + 2, np.shape(a)[1] + 2], 'float64')
    new_a[0:np.shape(a)[0], 0:np.shape(a)[1]] = a
    u, s, v = sparse.linalg.svds(new_a, k=min(new_a.shape[0], new_a.shape[1]) - 2)
    u = np.fliplr(u[:-2, :])
    s = s[::-1].real
    v = np.flipud(v[:, :-2])
    return u, s, v

def robust_svd(a: np.ndarray, tolerance: float = 0.0 * 1e-14) -> USV:
    try:
        u, s, v = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError as e1:
        logging.info(f'Ran into ({type(e1).__name__}: {e1})... transposing and retrying...')
        try:
            v, s, u = np.linalg.svd(np.transpose(a), full_matrices=False)
            u = np.transpose(u)
            v = np.transpose(v)
            logging.debug('Transposing succeeded!')
        except np.linalg.LinAlgError as e2:
            logging.warning(f'Ran into {type(e2).__name__}: {e2}... calling sparse code...')
            u, s, v = sparse_svd(a)
            logging.debug('Sparse code succeeded!')
    u = u[:, ~np.isclose(s, 0.0, atol=tolerance)]
    s = s[~np.isclose(s, 0.0, atol=tolerance)]
    v = v[np.transpose(~np.isclose(s, 0.0, atol=tolerance)), :]
    return u, s, v

def truncated_svd(tensor: np.ndarray, chi: int) -> USV:
    u, s, v = robust_svd(tensor)
    u = u[:, :chi]
    s = s[:chi]
    v = v[:chi, :]
    return u, s, v
