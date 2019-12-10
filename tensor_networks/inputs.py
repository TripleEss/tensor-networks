from __future__ import annotations
from tensor_networks.annotations import *

import numpy as np


class Input(ndarray, Sequence[ndarray]):
    def __new__(cls, arrays, label, **kwargs) -> Input:
        arr = np.array(arrays, **kwargs)
        self = arr.view(cls)
        self.label = label
        return self
