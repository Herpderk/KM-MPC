#!/usr/bin/env python3

import numpy as np
from numpy.random import uniform
from km_mpc.dynamics.vectors import State, Input, VectorList
from km_mpc.dynamics.models import CrazyflieModel, ParameterAffineQuadrotorModel
from km_mpc.utils.math import random_unit_quaternion
from km_mpc.optimization import NMPC


dt = 0.1
N = 50
model_nl = CrazyflieModel()
model = model_nl.as_affine()
Q = np.eye(model.nx)
R = np.eye(model.nu)
Qf = 2.0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=model, is_verbose=True)


x = State()
x.set_member('position_wf', uniform(-10.0, 10.0, size=3))
x.set_member('attitude', random_unit_quaternion())
x.set_member('linear_velocity_bf', uniform(-10.0, 10.0, size=3))
x.set_member('angular_velocity_bf', uniform(-10.0, 10.0, size=3))

lbu = Input(np.zeros(4))
ubu = Input(0.15 * np.ones(4))

xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )


nmpc.solve(x=x, xref=xref, uref=uref, lbu=lbu, ubu=ubu)
nmpc.plot_trajectory()
