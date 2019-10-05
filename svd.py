import logging
from typing import Tuple, Optional

import numpy as np


def truncate(u: np.ndarray, s: np.ndarray, v: np.ndarray, chi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if chi < 0:
        raise ValueError('chi has to be at least 0')
    if chi > s.size:
        # Pad with zeros
        return (
            np.pad(u, ((0, 0), (0, chi - u.shape[1]))),
            np.pad(s, (0, chi - s.size)),
            np.pad(v, ((0, chi - v.shape[0]), (0, 0))),
        )
    return u[:, :chi], s[:chi], v[:chi, :]


def truncated_svd(tensor: np.ndarray, chi: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, v = np.linalg.svd(tensor, full_matrices=False)
    _log_msg = f'{tensor.shape} --SVD--> {u.shape} {s.shape} {v.shape}'
    if chi is None:
        logging.debug(_log_msg)
        return u, s, v
    u, s, v = truncate(u, s, v, chi)
    _log_msg += f' --truncated/padded--> {u.shape} {s.shape} {v.shape}'
    logging.debug(_log_msg)
    return u, s, v
