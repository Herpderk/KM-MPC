#!/usr/bin/env python3

from km_mpc.dynamics.models import CrazyflieModel

from consts.trials import *
from consts.crazyflie import *
from consts.paths import DATA_PATH, CRAZYFLIE_PATH
from run_trials import run_trials_per_model


def main():
    nominal_model = CrazyflieModel(A)
    data_path = DATA_PATH + CRAZYFLIE_PATH
    run_trials_per_model(
        nominal_model=nominal_model,
        N=N, Q=Q, R=R, Qf=QF, M=M, P=P, S=S, P_aff=P_AFF, S_aff=S_AFF,
        lbp=LB_POS, ubp=UB_POS, lbv=LB_VEL, ubv=UB_VEL, lbw=LBW, ubw=UBW,
        lb_theta_factor=LB_THETA_FACTOR, ub_theta_factor=UB_THETA_FACTOR,
        data_path=data_path,
    )


if __name__=='__main__':
    main()
