import logging

import numpy as np

from tensor_networks.utils.annotations import *
from tensor_networks.utils.annotations import SVD


def truncated_svd(matrix: ndarray, chi: Optional[int] = None,
                  normalize: bool = True) -> SVD:
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    _log_msg = f'{matrix.shape} --SVD--> {u.shape} {s.shape} {v.shape}'
    if chi is None:
        pass
    elif chi < 0:
        raise ValueError('chi has to be at least 0')
    elif chi > s.size:
        # Pad with zeros
        u, s, v = (
            np.pad(u, ((0, 0), (0, chi - u.shape[1]))),
            np.pad(s, (0, chi - s.size)),
            np.pad(v, ((0, chi - v.shape[0]), (0, 0))),
        )
    else:
        u, s, v = u[:, :chi], s[:chi], v[:chi, :]
        if normalize:
            s = s * (np.linalg.norm(s) / np.linalg.norm(matrix))
    logging.debug(_log_msg + f' --truncated/padded-->'
                             f' {u.shape} {s.shape} {v.shape}')
    return u, s, v
