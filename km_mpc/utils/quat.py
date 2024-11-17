import casadi as cs
import numpy as np

from km_mpc.utils.math import skew


H = np.vstack((
    np.zeros((1,3)),
    np.eye(3),
))


def L_or_R(q: cs.SX, is_L: bool) -> cs.SX:
    assert q.shape[0] == 4
    scalar = q[0]
    vector = q[-3:]
    if not is_L:
        sign = -1
    else:
        sign = 1
    if type(q) == cs.SX:
        return cs.SX(cs.vertcat(
            cs.horzcat(scalar, -vector.T),
            cs.horzcat(vector, scalar * cs.SX.eye(3) + sign * cs.skew(vector))
        ))
    elif type(q) == np.ndarray:
        return np.vstack((
            np.hstack((scalar, -vector)),
            np.column_stack((vector, scalar * np.eye(3) + sign * skew(vector))),
        ))


def L(q: cs.SX) -> cs.SX:
    return L_or_R(q, is_L=True)


def R(q: cs.SX) -> cs.SX:
    return L_or_R(q, is_L=False)


def G(q: cs.SX) -> cs.SX:
    return L(q) @ H


def Q(q: cs.SX) -> cs.SX:
    return H.T @ R(q).T @ L(q) @ H
