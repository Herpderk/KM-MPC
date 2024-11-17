import casadi as cs
import numpy as np


def get_dimensions(config: dict, copies=1) -> int:
    dims = 0
    for config_member in config.values():
        dims += config_member['dimensions']
    return copies * dims


def symbolic(id: str, config: dict, copies=1) -> cs.SX:
    return cs.SX.sym(id, copies * config[id]['dimensions'])


def get_config_values(
    id: str,
    config: dict,
    copies: int = 1
) -> np.ndarray:
    dims = get_dimensions(config)
    i = 0
    vector = []
    for config_id in config.keys():
        delta_i = config[config_id]['dimensions']

        if delta_i == 1:
            vector += copies * [config[config_id][id]]
        elif delta_i > 1:
            vector += copies * list(config[config_id][id])

        i += delta_i
        if i > dims:
            raise IndexError
        elif i == dims:
            return np.array(vector)
    raise IndexError
