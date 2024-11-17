from typing import List

import casadi as cs
import numpy as np
from scipy.linalg import block_diag


def get_nominal_state_matrix(
    N: int,
) -> np.ndarray:
    A1 = np.zeros((3*(N-1), 3))
    A2 = np.eye(3*(N-1))
    A3 = np.zeros((3, 3*N))
    return np.vstack(( np.hstack((A1, A2)), A3) )


def get_attitude_state_matrix(N: int) -> cs.SX:
    return get_nominal_state_matrix(N)


def get_attitude_input_matrix(
    Hs: List[cs.SX],
    J: cs.SX,
) -> cs.SX:
    J_inv = cs.inv(J)
    N = len(Hs)
    B = cs.SX()
    for k in range(N):
        B = cs.vertcat( B, cs.horzcat( cs.SX.zeros(3,3), J_inv @ Hs[k] ) )
    return B


def get_translational_state_matrix(N: int) -> cs.SX:
    Ap = np.hstack((
        get_nominal_state_matrix(N), np.eye(3*N), np.zeros((3*N, 3*N)) ))
    Av = np.hstack((
        np.zeros((3*N, 3*N)), get_nominal_state_matrix(N), np.eye(3*N) ))
    Ag = np.hstack(( np.zeros((3*N, 2*3*N)), get_nominal_state_matrix(N) ))
    return np.vstack((Ap, Av, Ag))


def get_translational_input_matrix(
    Ps: List[cs.SX],
    Os: List[cs.SX],
    Vs: List[cs.SX],
    Gs: List[cs.SX],
    m: float,
) -> cs.SX:
    assert len(Ps) == len(Os) == len(Vs) == len(Gs)
    N = len(Ps)
    Bp = cs.SX()
    Bv = cs.SX()
    Bg = cs.SX()
    for k in range(N):
        Bp = cs.vertcat( Bp, cs.horzcat( cs.SX.zeros(3,3), -Ps[k] ) )
        Bv = cs.vertcat( Bv, cs.horzcat( Os[k]/m, -Vs[k] ) )
        Bg = cs.vertcat( Bg, cs.horzcat( cs.SX.zeros(3,3), -Gs[k] ) )
    return cs.vertcat(Bp, Bv, Bg)


def get_state_matrix(
    Np: int,
    Nw: int,
) -> np.ndarray:
    Ap = get_translational_state_matrix(Np)
    Aw = get_attitude_state_matrix(Nw)
    return block_diag(Aw, Ap)


def get_input_matrix(
    Ps: List[cs.SX],
    Os: List[cs.SX],
    Vs: List[cs.SX],
    Gs: List[cs.SX],
    Hs: List[cs.SX],
    J: cs.SX,
    m: float,
) -> cs.SX:
    B_translational = get_translational_input_matrix(Ps, Os, Vs, Gs, m)
    B_attitude = get_attitude_input_matrix(Hs, J)
    return cs.vertcat(B_translational, B_attitude)
