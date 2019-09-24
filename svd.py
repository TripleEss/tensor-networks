import logging
from typing import Tuple

import numpy as np


def truncated_svd(tensor: np.ndarray, chi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, v = np.linalg.svd(tensor, full_matrices=False)
    assert u.shape[1] == s.size == v.shape[0]

    if chi <= s.size:
        new_u = u[:, :chi]
        new_s = s[:chi]
        new_v = v[:chi, :]
    else:
        # Pad with zeros
        new_u = np.pad(u, ((0, 0), (0, chi - u.shape[1])))
        new_s = np.pad(s, (0, chi - s.size))
        new_v = np.pad(v, ((0, chi - v.shape[0]), (0, 0)))

    logging.debug(f'{tensor.shape} --SVD--> {u.shape} {s.shape} {v.shape}'
                  f' --truncated/padded--> '
                  f'{new_u.shape} {new_s.shape} {new_v.shape}')
    return new_u, new_s, new_v
