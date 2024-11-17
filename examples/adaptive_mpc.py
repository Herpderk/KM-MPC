#!/usr/bin/env python3

import numpy as np
from km_mpc.dynamics.vectors import State, Input, ProcessNoise, ModelParameters, \
                                    VectorList
from km_mpc.dynamics.models import CrazyflieModel, NonlinearQuadrotorModel
from km_mpc.utils.math import random_unit_quaternion
from km_mpc.optimization import NMPC, MHPE


# Parameter perturbation factors
lb_factor = 0.5
ub_factor = 1.5

# Perturb model parameters
model_inacc = CrazyflieModel(0.1 * np.ones(3))
param_nominal = model_inacc.parameters
perturb = np.random.uniform(low=lb_factor, high=ub_factor, size=model_inacc.ntheta)
param_perturb = ModelParameters(perturb * param_nominal.as_array())
model_acc = NonlinearQuadrotorModel(param_perturb, model_inacc.lbu, model_inacc.ubu)

# Init MPC
dt = 0.05
N = 25
Q = np.diag(np.hstack((
    2e0 * np.ones(3), 1e0 * np.ones(4), np.ones(3), 2e-1 * np.ones(3)
)))
R = 1e-1 * np.eye(4)
QF = 1e0* Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=QF, model=model_inacc)

# Init state
x = State()
x.set_member('position_wf', np.random.uniform(-10.0, 10.0, size=3))
x.set_member('attitude', random_unit_quaternion())
x.set_member('linear_velocity_bf', np.random.uniform(-10.0, 10.0, size=3))
x.set_member('angular_velocity_bf', np.random.uniform(-10.0, 10.0, size=3))

# Init MHPE
M = 10
P = np.diag(np.hstack((
    1.0e2,
    1.0e2 * np.ones(3),
    1.0e5 * np.ones(3),
    1.0e2 * np.ones(4),
    1.0e2 * np.ones(4),
    1.0e2 * np.ones(4)
)))
S = 1.0e6 * np.eye(13)
mhpe = MHPE(dt=dt, M=M, P=P, S=S, model=model_inacc, x0=x, plugin='ipopt')

# parameter bounds
lb_theta_init = lb_factor * param_nominal.as_array()
ub_theta_init = ub_factor * param_nominal.as_array()
lb_theta = ModelParameters(np.minimum(lb_theta_init, ub_theta_init))
ub_theta = ModelParameters(np.maximum(lb_theta_init, ub_theta_init))

# MPC args
theta = model_inacc.parameters
xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )
xs_guess = None
us_guess = None

# Sim stuff
w = ProcessNoise()
xs = VectorList()
us = VectorList()

# Iterate sim
sim_len = 200
for k in range(sim_len):
    # Solve, update warmstarts, and get the control input
    nmpc.solve(
        x=x, xref=xref, uref=uref, theta=theta,
        lbu=model_inacc.lbu, ubu=model_inacc.ubu,
        xs_guess=xs_guess, us_guess=us_guess
    )
    xs_guess = nmpc.get_predicted_states()
    us_guess = nmpc.get_predicted_inputs()
    u = us_guess.get(0)

    # Generate uniform noise on the acceleration
    lin_acc_noise = np.random.uniform(low=-10.0, high=10.0, size=3)
    ang_acc_noise = np.random.uniform(low=-10.0, high=10.0, size=3)
    w.set_member('linear_acceleration_bf', lin_acc_noise)
    w.set_member('angular_acceleration_bf', ang_acc_noise)

    # Update current state and trajectory history
    x = model_acc.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    # Get parameter estimate
    mhpe.solve(x, u, w)
    theta = mhpe.get_parameter_estimate()

    print(f'\ninput {k}: \n{u.as_array()}')
    print(f'\n\n\nstate {k+1}: \n{x.as_array()}')
    print(f'\nparameter estimate {k+1}: \n{theta.as_array()}')
print(f'\nnominal parameter: \n{model_inacc.parameters.as_array()}')
print(f'\ntrue parameter: \n{model_acc.parameters.as_array()}')

normalized_errors = np.zeros(model_acc.ntheta)
for i in range(model_acc.ntheta):
    theta_acc = model_acc.parameters.as_array()[i]
    normalized_errors[i] = ( theta.as_array()[i] - theta_acc ) / theta_acc

print(f'\nFinal parameter estimate error: {np.linalg.norm(normalized_errors)}')

nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_len)
