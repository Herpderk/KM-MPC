from km_mpc.dynamics.vectors import *
from km_mpc.dynamics.models import NonlinearQuadrotorModel
from km_mpc.optimization import NMPC, MHPE
from km_mpc.experiments.trials import run_parallel_trials
from km_mpc.experiments.random import *

from consts.trials import *


def run_trials_per_model(
    nominal_model: NonlinearQuadrotorModel,
    N: int,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    M: int,
    P: np.ndarray,
    S: np.ndarray,
    P_aff: np.ndarray,
    S_aff: np.ndarray,
    lbp: np.ndarray,
    ubp:np.ndarray,
    lbv: np.ndarray,
    ubv: np.ndarray,
    lbw: np.ndarray,
    ubw: np.ndarray,
    lb_theta_factor: float,
    ub_theta_factor: float,
    data_path: str,
) -> None:
    true_models = []
    random_states = []
    process_noises = []

    # Generate random sim conditions for all solvers
    for i in range(NUM_TRIALS):
        true_models += [
            get_random_model(nominal_model, lb_theta_factor, ub_theta_factor)]
        random_states += [
            get_random_state(lbp, ubp, lbv, ubv)]
        process_noises += [
            get_process_noise_seed(lbw, ubw, SIM_LEN)]

    lb_theta, ub_theta = get_parameter_bounds(
        nominal_model.parameters, lb_theta_factor, ub_theta_factor)

    for plugin, is_qp in SOLVERS.items():
            print('Starting trials for', plugin, '...')

            full_path = data_path + plugin + '/'

            if plugin == 'none':
                nmpc = NMPC(DT, N, Q, R, Qf, nominal_model)
                mhpe = None
            elif is_qp['is_qp']:
                nmpc = NMPC(DT, N, Q, R, Qf, nominal_model.as_affine())
                mhpe = MHPE(DT, M, P_aff, S_aff, nominal_model.as_affine(), plugin=plugin)
            else:
                nmpc = NMPC(DT, N, Q, R, Qf, nominal_model)
                mhpe = MHPE(DT, M, P, S, nominal_model, plugin=plugin)

            run_parallel_trials(
                is_affine=is_qp['is_qp'], data_path=full_path,
                dt=DT, nmpc=nmpc, mhpe=mhpe,
                lb_theta=lb_theta, ub_theta=ub_theta,
                nominal_model=nominal_model, true_models=true_models,
                random_states=random_states, process_noises=process_noises
            )

            print('Trials for', plugin, 'completed!')
