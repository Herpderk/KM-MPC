import numpy as np

from km_mpc.constants import BIG_NEGATIVE, BIG_POSITIVE, GRAVITY


PARAMETER_CONFIG = {
    'm': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'a': {
        'dimensions': 3,
        'lower_bound': np.zeros(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'Ixx': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Iyy': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Izz': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'b': {
        'dimensions': 4,
        'lower_bound': np.zeros(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'c': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'd': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
}


STATE_CONFIG = {
    'position_wf': {
        'dimensions': 3,
        'lower_bound': np.hstack((BIG_NEGATIVE * np.ones(3))),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'attitude': {
        'dimensions': 4,
        'lower_bound': -1.0 * np.ones(4),
        'upper_bound': 1.0 * np.ones(4),
        'default_value': np.hstack((1.0, np.zeros(3))),
    },
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'angular_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


KOOPMAN_STATE_CONFIG = {
    'position_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'gravity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.array([0.0, 0.0, -GRAVITY]),
    },
    'angular_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


INPUT_CONFIG = {
    'normalized_squared_motor_speed': {
        'dimensions': 4,
        'lower_bound': np.zeros(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
}


PROCESS_NOISE_CONFIG = {
    'linear_velocity_wf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'attitude_rate': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'linear_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'angular_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


KOOPMAN_PROCESS_NOISE_CONFIG = {
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'linear_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'gravity_rate_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.array([0.0, 0.0, -GRAVITY]),
    },
    'angular_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


NLP_SOLVER_CONFIG = {
    'ipopt': {
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-8,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.sb': 'yes',
    },
}


QP_SOLVER_CONFIG = {
    'osqp': {
        'osqp.max_iter': 4000,
        'osqp.eps_abs': 1e-8,
        'osqp.verbose': False,
    },
}
