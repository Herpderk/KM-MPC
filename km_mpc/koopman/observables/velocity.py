from typing import List

import casadi as cs

from km_mpc.utils.math import binomial_coefficient


def get_vs(
    v0: cs.SX,
    ws: List[cs.SX],
) -> List[cs.SX]:
    vs = [v0]
    N = len(ws)
    for k in range(1, N):
        vk = cs.SX.zeros(3)
        for n in range(k):
            vk += binomial_coefficient(k-1, n) * cs.skew(ws[n]).T @ vs[k-n-1]
        vs += [vk]
    return vs


def get_Os(
    vs: List[cs.SX]
) -> List[cs.SX]:
    N = len(vs)
    Os = [cs.SX.eye(3)]
    for k in range(1, N):
        Ok = cs.SX.zeros(3, 3)
        for n in range(k):
            Ok += binomial_coefficient(k-1, n) * cs.skew(vs[n-1]).T @ Os[k-n-1]
        Os += [Ok]
    return Os


def get_Vs(
    vs: List[cs.SX],
    ws: List[cs.SX],
    Hs: List[cs.SX],
    J: cs.SX,
) -> List[cs.SX]:
    assert len(vs) == len(ws) == len(Hs)
    N = len(vs)
    J_inv = cs.inv(J)
    Vs = [cs.SX.zeros((3,3))]
    for k in range(1, N):
        Vk = cs.SX.zeros((3,3))
        for n in range(k):
            Vk += binomial_coefficient(k-1, n) * cs.skew(vs[n]).T @ J_inv @ Hs[k-n-1]
            Vk += binomial_coefficient(k-1, n) * cs.skew(ws[n]).T @ J_inv @ Vs[k-n-1]
        Vs += [Vk]
    return Vs
