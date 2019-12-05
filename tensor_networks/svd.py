import logging
from functools import partial

import numpy as np

from tensor_networks.annotations import *


standard_svd = partial(np.linalg.svd, full_matrices=False)


def truncated_svd(matrix: ndarray, *,
                  compute_chi: Optional[Callable[SVD, int]] = None,
                  max_chi: Optional[int] = None,
                  normalize: bool = True) -> SVDTuple:
    u, s, v = standard_svd(matrix)

    if compute_chi:
        max_chi = min(max_chi, compute_chi(u, s, v))

    if max_chi is None or len(s) < max_chi:
        new_u, new_s, new_v = u, s, v
    elif max_chi < 1:
        raise ValueError('max_chi has to be at least 1')
    else:
        new_u, new_s, new_v = u[:, :max_chi], s[:max_chi], v[:max_chi, :]
        if normalize:
            s = s * (np.linalg.norm(new_s) / np.linalg.norm(s))

    logging.debug(f'{matrix.shape} --SVD--> {u.shape} {s.shape} {v.shape}'
                  f' --truncated--> {new_u.shape} {new_s.shape} {new_v.shape}')
    return new_u, new_s, new_v
