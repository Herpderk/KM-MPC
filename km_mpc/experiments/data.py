from typing import Union, List, Tuple

import numpy as np

from km_mpc.dynamics.vectors import *


class TrialData():
    def __init__(
        self,
        x: State,
        u: Input,
        w: ProcessNoise,
        theta: Union[ModelParameters, AffineModelParameters],
        xref: State,
        uref: Input,
        theta_true: Union[ModelParameters, AffineModelParameters],
        Q: np.ndarray,
        R: np.ndarray,
        mhpe_solution: dict,
        mhpe_solver_stats: dict,
    ) -> None:
        self.x = x
        self.u = u
        self.w = w
        self.theta = theta
        self.xref = xref
        self.uref = uref
        self.theta_true = theta_true
        self.Q = Q
        self.R = R
        self.mhpe_solution = mhpe_solution
        self.mhpe_solver_stats = mhpe_solver_stats


def get_mhpe_solve_times(dataset: List[TrialData]) -> np.ndarray:
    try:
        return np.array([data_k.mhpe_solver_stats['t_wall_total'] for data_k in dataset])
    except KeyError:
        return np.array([data_k.mhpe_solver_stats['t_wall_solver'] for data_k in dataset])


def get_average_mhpe_solve_time(dataset: List[TrialData]) -> float:
    return np.average(get_mhpe_solve_times(dataset))


def get_mhpe_solve_time_quartiles(dataset: List[TrialData]) -> Tuple[float]:
    solve_times = get_mhpe_solve_times(dataset)
    Q1 = np.quantile(solve_times, 0.25)
    Q2 = np.quantile(solve_times, 0.50)
    Q3 = np.quantile(solve_times, 0.75)
    return Q1, Q2, Q3


def get_mhpe_non_convergence_rate(dataset: List[TrialData]) -> float:
    non_convergences = 0
    for data_k in dataset:
        if not data_k.mhpe_solver_stats['success']:
            non_convergences += 1
    return non_convergences / len(dataset)


def get_states(dataset: List[TrialData]) -> VectorList:
    return VectorList([data_k.x for data_k in dataset])


def get_inputs(dataset: List[TrialData]) -> VectorList:
    return VectorList([data_k.u for data_k in dataset])


def get_cost(dataset: List[TrialData]) -> float:
    xs = get_states(dataset).as_array()
    us = get_inputs(dataset).as_array()
    xrefs = [data_k.xref.as_array() for data_k in dataset]
    urefs = [data_k.uref.as_array() for data_k in dataset]
    Qs = [data_k.Q for data_k in dataset]
    Rs = [data_k.R for data_k in dataset]
    cost = 0.0
    for i in range(len(dataset)):
        x_err = xs[i] - xrefs[i][0, :]
        u_err = us[i] - urefs[i][0, :]
        cost += x_err.T @ Qs[i] @ x_err + u_err.T @ Rs[i] @ u_err
    return cost
