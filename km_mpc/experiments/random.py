import numpy as np

from km_mpc.dynamics.vectors import *
from km_mpc.dynamics.models import NonlinearQuadrotorModel
from km_mpc.utils.math import random_unit_quaternion


def get_process_noise_seed(
    lbw: np.ndarray,
    ubw: np.ndarray,
    sim_len: int,
) -> VectorList:
    ws = VectorList()
    for i in range(sim_len):
        w_arr = np.random.uniform(lbw, ubw)
        ws.append(ProcessNoise(w_arr))
    return ws


def get_random_state(
    lb_pos: np.ndarray,
    ub_pos: np.ndarray,
    lb_vel: np.ndarray,
    ub_vel: np.ndarray,
) -> State:
    x = State()
    x.set_member('position_wf', np.random.uniform(lb_pos, ub_pos))
    x.set_member('attitude', random_unit_quaternion())
    x.set_member('linear_velocity_bf', np.random.uniform(lb_vel, ub_vel))
    x.set_member('angular_velocity_bf', np.random.uniform(lb_vel, ub_vel))
    return x


def get_random_model(
    nominal_model: NonlinearQuadrotorModel,
    lb_factor: float,
    ub_factor: float,
) -> NonlinearQuadrotorModel:
    perturb = np.random.uniform(lb_factor, ub_factor, nominal_model.ntheta)
    param_perturb = ModelParameters(perturb * nominal_model.parameters.as_array())
    return NonlinearQuadrotorModel(param_perturb, nominal_model.lbu, nominal_model.ubu)
