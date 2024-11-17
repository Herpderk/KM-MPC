from typing import List

import casadi as cs

from km_mpc.utils.math import binomial_coefficient


def get_ps(
    p0: cs.SX,
    ws: List[cs.SX]
) -> List[cs.SX]:
    N = len(ws)
    ps = [p0]
    for k in range(1, N):
        pk = cs.SX.zeros(3)
        for n in range(k):
            pk += binomial_coefficient(k-1, n) * cs.skew(ws[n]).T @ ps[k-n-1]
        ps += [pk]
    return ps


def get_Ps(
    ps: List[cs.SX],
    ws: List[cs.SX],
    Hs: List[cs.SX],
    J: cs.SX,
) -> List[cs.SX]:
    assert len(ps) == len(ws) == len(Hs)
    N = len(ps)
    J_inv = cs.inv(J)
    Ps = [cs.SX.zeros((3,3))]
    for k in range(1, N):
        Pk = cs.SX.zeros((3,3))
        for n in range(k):
            Pk += binomial_coefficient(k-1, n) * \
                cs.skew(ps[n]).T @ J_inv @ Hs[k-n-1]
            Pk += binomial_coefficient(k-1, n) * \
                cs.skew(ws[n]).T @ J_inv @ Ps[k-n-1]
        Ps += [Pk]
    return Ps
