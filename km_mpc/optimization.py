from typing import List, Union
import time

import casadi as cs
import matplotlib.ticker
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams['text.usetex'] = True

from km_mpc.config import *
from km_mpc.dynamics.models import *
from km_mpc.dynamics.vectors import *
from km_mpc.utils.config import get_config_values, get_dimensions
from km_mpc.utils.misc import is_none


class NMPC():
    def __init__(
        self,
        dt: float,
        N: int,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        model: DynamicsModel,
        is_verbose=False,
    ) -> None:
        self._dt = dt
        self._N = N
        self._Q = Q
        self._R = R
        self._Qf = Qf
        self._model = model
        self._sol = {}
        self._lbg = None
        self._ubg = None

        if type(model) == KoopmanLiftedQuadrotorModel:
            self._is_koopman = True
            self._state_class = KoopmanLiftedState
        else:
            self._is_koopman = False
            self._state_class = State

        self._lbx = self._state_class(
            get_config_values('lower_bound', model.state_config, model.order),
            self._model.order,
        )
        self._ubx = self._state_class(
            get_config_values('upper_bound', model.state_config, model.order),
            self._model.order,
        )
        self._lbu = Input(get_config_values('lower_bound', INPUT_CONFIG))
        self._ubu = Input(get_config_values('upper_bound', INPUT_CONFIG))
        self._theta = self._model.parameters
        self._us_guess = VectorList(self._N * [self._model.ubu])
        self._solver = self._init_solver(is_verbose)

    @property
    def N(self) -> int:
        return self._N

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @property
    def R(self) -> np.ndarray:
        return self._R

    def get_predicted_states(self) -> VectorList:
        nx = self._model.nx
        nu = self._model.nu
        xs = VectorList()
        for k in range(1, self._N + 1):
            if self._is_koopman:
                xs.append(KoopmanLiftedState(np.array(
                    self._sol['x'][k*(nx+nu) : k*(nx+nu) + nx]).flatten(),
                    self._model.order
                ))
            else:
                xs.append(State(np.array(
                    self._sol['x'][k*(nx+nu) : k*(nx+nu) + nx]).flatten()))
        return xs

    def get_predicted_inputs(self) -> VectorList:
        nx = self._model.nx
        nu = self._model.nu
        us = VectorList()
        for k in range(self._N):
            us.append(Input(np.array(
                self._sol['x'][(k+1)*nx+k*nu : (k+1)*nx+(k+1)*nu]).flatten()))
        return us

    def plot_trajectory(
        self,
        xs: VectorList = None,
        us: VectorList = None,
        dt: float = None,
        N: float = None,
        order: int = None,
    ) -> None:
        '''
        Display the series of control inputs
        and trajectory over prediction horizon.
        '''
        if is_none(dt): dt = self._dt
        if is_none(N): N = self._N

        t = dt * np.arange(N)
        interp_N = 1000
        fig, axs = plt.subplots(5, figsize=(6, 6))

        if is_none(us):
            us = self.get_predicted_inputs().as_array()
        else:
            us = us.as_array()
        legend = [r'$u_1(t)$', r'$u_2(t)$', r'$u_3(t)$', r'$u_4(t)$']
        self._plot_trajectory(
            axs[0], t, us, interp_N, legend,
            'Thrust (N)',
        )

        if not self._is_koopman:
            if is_none(xs):
                xs = self.get_predicted_states().as_array()
            else:
                xs = xs.as_array()
            if len(xs) > len(us):
                xs = xs[:len(us), :]
            legend = [r'$p_x^W(t)$', r'$p_y^W(t)$', r'$p_z^W(t)$']
            self._plot_trajectory(
                axs[1], t, xs[:,:3], interp_N, legend,
                'Position (m)'
            )
            legend = [r'$q_w^W(t)$', r'$q_x^W(t)$', r'$q_y^W(t)$', r'$q_z^W(t)$']
            self._plot_trajectory(
                axs[2], t, xs[:, 3:7], interp_N, legend,
                'Attitude (quat)'
            )
            legend = [r'$v_x^B(t)$', r'$v_y^B(t)$', r'$v_z^B(t)$']
            self._plot_trajectory(
                axs[3], t, xs[:, 7:10], interp_N, legend,
                'Linear\nvelocity (m/s)'
            )
            legend = [r'$w_x^B(t)$', r'$w_y^B(t)$', r'$w_z^B(t)$']
            self._plot_trajectory(
                axs[4], t, xs[:, 10:13], interp_N, legend,
                'Angular\nvelocity (rad/s)',
            )

        for ax in axs.flat:
            ax.set(xlabel='Time (s)')
            ax.label_outer()
        plt.show()

    def solve(
        self,
        x: Union[State, KoopmanLiftedState],
        xref: VectorList,
        uref: VectorList,
        theta: ModelParameters = None,
        lbx: State = None,
        ubx: State = None,
        lbu: Input = None,
        ubu: Input = None,
        xs_guess: VectorList = None,
        us_guess: VectorList = None,
    ) -> dict:
        # Enforce correct horizon length
        assert len(xref.get()) == len(uref.get()) == self._N

        # Get default inequality constraints
        if is_none(lbx): lbx = self._lbx
        if is_none(ubx): ubx = self._ubx
        if is_none(lbu): lbu = self._lbu
        if is_none(ubu): ubu = self._ubu
        if is_none(theta): theta = self._theta
        if is_none(us_guess): us_guess = self._us_guess
        if is_none(xs_guess): xs_guess = xref #  VectorList(self._N * [x])

        # Initialize the parameter argument
        p = theta.as_list()
        if self._is_koopman:
            assert type(x) == KoopmanLiftedState
            p += KoopmanLiftedState(x.get_zero_order_array(), 1).as_list()
        else:
            assert type(x) == State

        # Construct optimization arguments
        lbd = x.as_list()
        ubd = x.as_list()
        guess = x.as_list()
        for k in range(self._N):
            lbd += lbu.as_list() + lbx.as_list()
            ubd += ubu.as_list() + ubx.as_list()
            guess += us_guess.get(k).as_list() + xs_guess.get(k).as_list()
            p += uref.get(k).as_list() + xref.get(k).as_list()

        # Solve
        self._sol = self._solver(
            x0=guess, p=p, lbx=lbd, ubx=ubd, lbg=self._lbg, ubg=self._ubg)
        return self._sol

    def _init_solver(
        self,
        is_verbose: bool
    ) -> dict:
        # Decision variable for state
        x0 = cs.SX.sym('x0', self._model.nx)
        # Constant for model parameters
        theta = cs.SX.sym('theta', self._model.ntheta)
        # Constant for koopman initialization
        if self._is_koopman:
            z0 = cs.SX.sym('z0', get_dimensions(KOOPMAN_STATE_CONFIG))
        else:
            z0 = cs.SX()

        # Variables for formulating NLP
        p = [theta, z0]
        d = [x0]
        g = []
        lbg = []
        ubg = []
        J = 0.0

        # Formulate the nlp
        xk = x0
        for k in range(self._N):
            # New decision variable for control
            uk = cs.SX.sym('u' + str(k), self._model.nu)
            d += [uk]

            # Get the state at the end of the time step
            if self._is_koopman:
                xf = self._model.F(self._dt, x=xk, u=uk, theta=theta, z0=z0)
            else:
                xf = self._model.F(dt=self._dt, x=xk, u=uk, theta=theta)

            # New NLP variable for state at the end of the interval
            xk = cs.SX.sym('x' + str(k+1), self._model.nx)
            d += [xk]

            # New constants for reference tracking
            uref_k = cs.SX.sym('uref' + str(k), self._model.nu)
            xref_k = cs.SX.sym('xref' + str(k+1), self._model.nx)
            p += [uref_k, xref_k]

            # Add costs
            J += self._get_stage_cost(x=xk, u=uk, xref=xref_k, uref=uref_k)
            if k == self._N - 1:
                J += self._get_terminal_cost(x=xk, xref=xref_k)

            # Add dynamics equality constraint
            g += [xf - xk]
            lbg += self._model.nx * [0.0]
            ubg += self._model.nx * [0.0]

        # Concatenate decision variables and constraint terms
        d = cs.vertcat(*d)
        p = cs.vertcat(*p)
        g = cs.vertcat(*g)

        # Initialize equality constraint values
        self._lbg = lbg
        self._ubg = ubg

        # Create NLP solver
        nlp_prob = {'f': J, 'x': d, 'p': p, 'g': g}
        opts = {
            'error_on_fail': False,
            'ipopt.max_iter': 3000,
            'ipopt.tol': 1e-8,
        }
        if not is_verbose:
            opts['ipopt.print_level'] = 0
            opts['print_time'] = 0
            opts['ipopt.sb'] = 'yes'
        return cs.nlpsol('nlp_solver', 'ipopt', nlp_prob, opts)

    def _get_stage_cost(
        self,
        x: cs.SX,
        u: cs.SX,
        xref: cs.SX,
        uref: cs.SX,
    ) -> cs.SX:
        state_err = x - xref
        input_err = u - uref
        return state_err.T @ self._Q @ state_err + \
                input_err.T @ self._R @ input_err

    def _get_terminal_cost(
        self,
        x: cs.SX,
        xref: np.ndarray,
    ) -> cs.SX:
        err = x - xref
        return err.T @ (self._Qf - self._Q) @ err

    def _plot_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        Xs: np.ndarray,
        traj: np.ndarray,
        interp_N: int,
        legend: List[str],
        ylabel: str,
    ) -> None:
        formatter = matplotlib.ticker.FormatStrFormatter('%.1f')
        ax.yaxis.set_major_formatter(formatter)
        ax.locator_params(axis='y', nbins=3)
        ax.set_ylabel(ylabel, loc='bottom')
        for i in range(traj.shape[1]):
            x_interp = self._get_interpolation(Xs, Xs, interp_N)
            y_interp = self._get_interpolation(Xs, traj[:, i], interp_N)
            ax.plot(x_interp, y_interp, label=legend[i])
        ax.legend(loc='upper right')

    def _get_interpolation(
        self,
        Xs: np.ndarray,
        Ys: np.ndarray,
        N: int,
    ) -> np.ndarray:
        spline_func = make_interp_spline(Xs, Ys)
        interp_x = np.linspace(Xs.min(), Xs.max(), N)
        interp_y = spline_func(interp_x)
        return interp_y


class MHPE():
    def __init__(
        self,
        dt: float,
        M: int,
        P: np.ndarray,
        S: np.ndarray,
        model: DynamicsModel,
        x0: State = State(),
        plugin: str = 'ipopt',
    ) -> None:
        assert type(model) == NonlinearQuadrotorModel \
            or type(model) == ParameterAffineQuadrotorModel
        self._dt = dt
        self._M = M
        self._P = P
        self._S = S
        self._model = model
        self._plugin = plugin

        self._solve_time = 0.0
        self._sol = {}
        self._xs = VectorList(x0)
        self._us = VectorList()
        self._ws = VectorList()
        self._theta = self._model.parameters

        if type(model) == ParameterAffineQuadrotorModel:
            self._parameter_class = AffineModelParameters
        else:
            self._parameter_class = ModelParameters

        self._lb_theta = self._parameter_class(get_config_values(
            'lower_bound', model.parameters.config))
        self._ub_theta = self._parameter_class(get_config_values(
            'upper_bound', model.parameters.config))
        self._lbw = ProcessNoise(get_config_values(
            'lower_bound', PROCESS_NOISE_CONFIG))
        self._ubw = ProcessNoise(get_config_values(
            'upper_bound', PROCESS_NOISE_CONFIG))
        self._lbg = []
        self._ubg = []
        self._solver = self._init_solver()

    @property
    def M(self) -> int:
        return self._M

    def reset_measurements(self, x0: State) -> None:
        self._xs = VectorList(x0)
        self._ws = VectorList()

    def get_solver_stats(self) -> dict:
        stats = self._solver.stats()
        return self._fix_solver_stats(stats)

    def get_full_solution(self) -> dict:
        return self._sol

    def get_parameter_estimate(self) -> Union[ModelParameters, AffineModelParameters]:
        return self._theta

    def get_process_noise_estimates(self) -> VectorList:
        return self._ws

    def solve(
        self,
        xM: State,
        uM: Input,
        wM: ProcessNoise = ProcessNoise(),
        lb_theta: Union[ModelParameters, AffineModelParameters] = None,
        ub_theta: Union[ModelParameters, AffineModelParameters] = None,
        lbw: ProcessNoise = None,
        ubw: ProcessNoise = None,
        theta_guess: ModelParameters = None,
        ws_guess: VectorList = None,
    ) -> dict:
        # Update measurement history
        self._update_measurements(xM, uM, wM)

        # Skip this solver call if measurement history isn't full length
        if not self._measurements_are_full():
            return self._sol

        # Get default inequality constraints
        if is_none(lb_theta): lb_theta = self._lb_theta
        if is_none(ub_theta): ub_theta = self._ub_theta
        if is_none(lbw): lbw = self._lbw
        if is_none(ubw): ubw = self._ubw
        if is_none(theta_guess): theta_guess = self._theta
        if is_none(ws_guess): ws_guess = self._ws

        # Construct optimization arguments
        guess = theta_guess.as_list() + list(ws_guess.as_array().flatten())
        lbd = lb_theta.as_list()
        ubd = ub_theta.as_list()
        p = self._xs.get(0).as_list() + self._theta.as_list()
        for k in range(self._M):
            lbd += lbw.as_list()
            ubd += ubw.as_list()
            p += self._us.get(k).as_list() + self._xs.get(k + 1).as_list() \
                    + self._ws.get(k).as_list()

        # Solve
        st = time.perf_counter()
        self._sol = self._solver(
            x0=guess, p=p, lbx=lbd, ubx=ubd, lbg=self._lbg, ubg=self._ubg)
        et = time.perf_counter()
        self._solve_time = et - st

        self._update_estimates()
        return self._sol

    def _init_solver(self) -> dict:
        # Constant for state
        x0 = cs.SX.sym('x0', self._model.nx)
        # Decision variable for model parameters
        theta = cs.SX.sym('theta', self._model.ntheta)
        # Constant for parameter reference
        theta_ref = cs.SX.sym('theta_ref', self._model.ntheta)

        # Arguments for formulating NLP
        p = [x0, theta_ref]
        d = [theta]
        g = []
        lbg = []
        ubg = []
        J = self._get_parameter_cost(theta, theta_ref)

        # Formulate the NLP
        xk = x0
        for k in range(self._M):
            # New constant for control input
            uk = cs.SX.sym('u' + str(k), self._model.nu)
            p += [uk]

            # New decision variable for process noise
            wk = cs.SX.sym('w' + str(k), self._model.nw)
            d += [wk]

            # Get the state at the end of the time step
            xf = self._model.F_euler(dt=self._dt, x=xk, u=uk, w=wk, theta=theta)

            # New constant for the state at the end of the interval
            xk = cs.SX.sym('x' + str(k+1), self._model.nx)
            p += [xk]

            # New constant for disturbance reference tracking
            wref_k = cs.SX.sym('wref' + str(k), self._model.nw)
            p += [wref_k]

            # Add running cost
            J += self._get_stage_cost(wk, wref_k)

            # Add dynamics equality constraint
            g += [xf - xk]
            lbg += self._model.nx * [0.0]
            ubg += self._model.nx * [0.0]

        # Concatenate decision variables and constraint terms
        d = cs.vertcat(*d)
        p = cs.vertcat(*p)
        g = cs.vertcat(*g)

        # Initialize equality constraint values
        self._lbg = lbg
        self._ubg = ubg

        # Create NLP solver
        opts = self._get_solver_opts()
        nlp_prob = {'f': J, 'x': d, 'p': p, 'g': g}
        if self._is_qp():
            return cs.qpsol('qp_solver', self._plugin, nlp_prob, opts)
        else:
            return cs.nlpsol('nlp_solver', self._plugin, nlp_prob, opts)

    def _get_solver_opts(self) -> dict:
        opts = {'error_on_fail': False}
        try:
            solver_config = QP_SOLVER_CONFIG[self._plugin]
        except KeyError:
            solver_config = NLP_SOLVER_CONFIG[self._plugin]
        finally:
            for key in solver_config.keys():
                opts[key] = solver_config[key]
        return opts

    def _fix_solver_stats(self, stats: dict) -> dict:
        try:
            if stats['t_wall_solver'] <= 0.0:
                stats['t_wall_solver'] = self._solve_time
        except KeyError:
            try:
                if stats['t_wall_total'] <= 0.0:
                    stats['t_wall_total'] = self._solve_time
            except KeyError:
                stats['t_wall_total'] = self._solve_time
        return stats

    def _measurements_are_full(
        self
    ) -> bool:
        if len(self._xs.get()) < self._M+1 or len(self._us.get()) < self._M \
        or len(self._ws.get()) < self._M:
            return False
        else:
            return True

    def _update_measurements(
        self,
        xM: State,
        uM: State,
        wM: ProcessNoise,
    ) -> None:
        if self._measurements_are_full():
            self._xs.pop(0)
            self._us.pop(0)
            self._ws.pop(0)
        self._xs.append(xM)
        self._us.append(uM)
        self._ws.append(wM)

    def _update_estimates(self):
        self._theta = self._parameter_class(
            np.array(self._sol['x'][:self._model.ntheta]).flatten())
        ws = []
        for i in range(self._M):
            wk = self._sol['x'][self._model.ntheta + i*self._model.nw : \
                                self._model.ntheta + (i+1)*self._model.nw]
            ws += [ProcessNoise(np.array(wk).flatten())]
        self._ws = VectorList(ws)

    def _get_stage_cost(
        self,
        w: cs.SX,
        wref: cs.SX,
    ) -> cs.SX:
        err = w - wref
        return err.T @ self._S @ err

    def _get_parameter_cost(
        self,
        theta: cs.SX,
        theta_ref: cs.SX,
    ) -> cs.SX:
        err = theta - theta_ref
        return err.T @ self._P @ err

    def _is_qp(
        self,
    ) -> bool:
        try:
            QP_SOLVER_CONFIG[self._plugin]
            return True
        except KeyError:
            return False
        except TypeError:
            return False
