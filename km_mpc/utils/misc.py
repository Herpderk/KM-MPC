from typing import Any, Union, Tuple

import numpy as np
import casadi as cs


def is_none(a: Any) -> bool:
    if type(a) == type(None):
        return True
    else:
        return False


def alternating_ones(shape: Union[int, Tuple[int]]) -> np.ndarray:
    ones = np.ones(shape)
    ones[::2] = -1.0
    return ones


def convert_casadi_to_numpy_array(a: Union[cs.SX, cs.DM]) -> np.ndarray:
    return np.array(cs.DM(a)).flatten()
