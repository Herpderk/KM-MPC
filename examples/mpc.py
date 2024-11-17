#!/usr/bin/env python3

import numpy as np
from numpy.random import uniform
from km_mpc.dynamics.vectors import State, Input, ProcessNoise, VectorList
from km_mpc.dynamics.models import CrazyflieModel, FusionOneModel
from km_mpc.utils.math import random_unit_quaternion
from km_mpc.optimization import NMPC


dt = 0.02
N = 25
model = FusionOneModel(a=np.array([0.01, 0.01, 0.05]))
Q = np.diag(np.hstack((
    2e0 * np.ones(3), 1e0 * np.ones(4), np.ones(3), 2e-1 * np.ones(3)
)))
R = 1e-2 * np.eye(model.nu)
Qf = 1e0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=model, is_verbose=False)

x = State()
x.set_member('position_wf', uniform(-10.0, 10.0, size=3))
x.set_member('attitude', random_unit_quaternion())
x.set_member('linear_velocity_bf', uniform(-10.0, 10.0, size=3))
x.set_member('angular_velocity_bf', uniform(-10.0, 10.0, size=3))

w = ProcessNoise()

xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )

xs = VectorList()
us = VectorList()

sim_length = 250
for k in range(sim_length):
    # Solve, update warmstarts, and get the control input
    nmpc.solve(
        x=x, xref=xref, uref=uref, lbu=model.lbu, ubu=model.ubu,
    )
    u = nmpc.get_predicted_inputs().get(0)

    # Generate Guassian noise on the second order terms
    lin_acc_noise = np.random.uniform(-2.5 * np.ones(3), 2.5 * np.ones(3))
    ang_acc_noise = np.random.uniform(-2.5 * np.ones(3), 2.5 * np.ones(3))
    w.set_member('linear_acceleration_bf', lin_acc_noise)
    w.set_member('angular_acceleration_bf', ang_acc_noise)

    # Update current state and trajectory history
    x = model.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    print(f'\ninput {k}: \n{u.as_array()}')
    print(f'\n\n\nstate {k+1}: \n{x.as_array()}')

nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_length)
