from typing import List

import casadi as cs

from km_mpc.utils.math import binomial_coefficient


def get_gs(
    g0: cs.SX,
    ws: List[cs.SX],
) -> List[cs.SX]:
    N = len(ws)
    gs = [g0]
    for k in range(1, N):
        gk = cs.SX.zeros(3)
        for n in range(k):
            gk += binomial_coefficient(k-1, n) * cs.skew(ws[n]).T @ gs[k-n-1]
        gs += [gk]
    return gs


def get_Gs(
    gs: List[cs.SX],
    ws: List[cs.SX],
    Hs: List[cs.SX],
    J: cs.SX,
) -> List[cs.SX]:
    assert len(gs) == len(ws) == len(Hs)
    N = len(gs)
    J_inv = cs.inv(J)
    Gs = [cs.SX.zeros((3,3))]
    for k in range(1, N):
        Gk = cs.SX.zeros((3,3))
        for n in range(k):
            Gk += binomial_coefficient(k-1, n) * \
                cs.skew(gs[n]).T @ J_inv @ Hs[k-n-1]
            Gk += binomial_coefficient(k-1, n) * \
                cs.skew(ws[n]).T @ J_inv @ Gs[k-n-1]
        Gs += [Gk]
    return Gs
