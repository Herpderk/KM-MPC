from typing import List

import casadi as cs

from km_mpc.utils.math import binomial_coefficient


def gamma(
    J: cs.SX,
    w: cs.SX,
) -> cs.SX:
    return J @ w


def h(
    J: cs.SX,
    w: cs.SX,
) -> cs.SX:
    return cs.skew(gamma(J, w)) @ cs.inv(J) - cs.skew(w)


def get_ws(
    w0: cs.SX,
    J: cs.SX,
    N: int,
) -> List[cs.SX]:
    J_inv = cs.inv(J)
    ws = [cs.SX(w0)]
    for k in range(1, N):
        summation = cs.SX.zeros(3)
        for n in range(k):
            gamma_n = gamma(J, ws[n])
            summation += binomial_coefficient(k-1, n) * \
                        cs.skew(gamma_n) @ ws[k-n-1]
        wk = J_inv @ summation
        ws += [wk]
    return ws


def get_Hs(
    ws: List[cs.SX],
    J: cs.SX,
) -> List[cs.SX]:
    N = len(ws)
    Hs = [cs.SX.eye(3)]
    for k in range(1, N):
        Hk = cs.SX.zeros((3,3))
        for n in range(k):
            Hk += binomial_coefficient(k-1, n) * h(J, ws[n]) @ Hs[k-n-1]
        Hs += [Hk]
    return Hs
