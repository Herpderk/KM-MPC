import numpy as np


# NMPC stuff
N = 25
Q = np.diag(np.hstack((
    2e0 * np.ones(3), 1e0 * np.ones(4), np.ones(3), 2e-1 * np.ones(3)
)))
R = 1e-1 * np.eye(4)
QF = 1e0* Q


# Nonlinear MHE weights
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


# Linear-quadratic MHE weights
P_AFF = np.diag(np.hstack((
    1.0e-1,
    1.0e1 * np.ones(2), 1.0e0,
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e1 * np.ones(2), 1.0e2,
)))
S_AFF = 1.0e6 * np.eye(13)


LB_POS = -5.0 * np.ones(3)
UB_POS = 5.0 * np.ones(3)


LB_VEL = -2.5 * np.ones(3)
UB_VEL = 2.5 * np.ones(3)


# process noise constants
LBW = np.hstack((np.zeros(7), -2.5 * np.ones(6)))
UBW = np.hstack((np.zeros(7),  2.5 * np.ones(6)))


# Aerodynamic drag
A = np.array([0.01, 0.01, 0.05])


# Parameter uncertainty bounds
LB_THETA_FACTOR = 0.5
UB_THETA_FACTOR = 1.5
