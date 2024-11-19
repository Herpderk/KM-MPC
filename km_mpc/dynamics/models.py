from typing import Union, Callable

import casadi as cs
import numpy as np

from km_mpc.utils import quat
from km_mpc.dynamics.vectors import *
from km_mpc.koopman.dynamics import get_state_matrix, get_input_matrix
from km_mpc.koopman.observables import attitude, gravity, velocity, position
from km_mpc.constants import GRAVITY
from km_mpc.utils.misc import is_none, alternating_ones
from km_mpc.utils.config import symbolic, get_dimensions, get_config_values
from km_mpc.config import *


class DynamicsModel():
    def __init__(
        self,
        parameters: ModelParameters,
        lbu: Input,
        ubu: Input,
        state_config: dict,
        input_config: dict,
        noise_config: dict,
        order: int,
    ) -> None:
        self._f = None
        self._parameters = parameters
        self._state_config = state_config
        self._input_config = input_config
        self._noise_config = noise_config
        self._order = order
        self._lbu = lbu
        self._ubu = ubu

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    @parameters.setter
    def parameters(
        self,
        parameters: ModelParameters
    ) -> None:
        self._parameters = parameters

    @property
    def nx(self) -> int:
        return get_dimensions(self._state_config, self._order)

    @property
    def nw(self) -> int:
        return get_dimensions(self._noise_config, self._order)

    @property
    def nu(self) -> int:
        return get_dimensions(self._input_config)

    @property
    def ntheta(self) -> int:
        return get_dimensions(self._parameters.config)

    @property
    def b(self) -> Input:
        return self._b

    @property
    def lbu(self) -> Input:
        return self._lbu

    @property
    def ubu(self) -> Input:
        return self._ubu

    @property
    def state_config(self) -> dict:
        return self._state_config

    @property
    def noise_config(self) -> dict:
        return self._noise_config

    @property
    def parameter_config(self) -> dict:
        return self._parameter_config

    @property
    def order(self)-> int:
        return self._order

    def step_sim(
        self,
        dt: float,
        x: State,
        u: Input = None,
        w: ProcessNoise = None,
        theta: ModelParameters = None,
    ) -> State:
        if is_none(u): u = Input()
        if is_none(w): w = ProcessNoise()
        if is_none(theta): theta = self._parameters
        xf = self.F(
            dt, x.as_array(), u.as_array(), w.as_array(), theta.as_array())
        return State(xf)

    def F(
        self,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX] = None,
        theta: Union[np.ndarray, cs.SX] = None,
    ) -> Union[np.ndarray, cs.SX]:
        xf = self.rk4(self.f, dt, x, u, w, theta)
        if type(xf) == cs.DM:
            return np.array(xf).flatten()
        else:
            return xf

    def F_euler(
        self,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX] = None,
        theta: Union[np.ndarray, cs.SX] = None,
    ) -> Union[np.ndarray, cs.SX]:
        xf = self.forward_euler(self.f, dt, x, u, w, theta)
        if type(xf) == cs.DM:
            return np.array(xf).flatten()
        else:
            return xf

    def f(
        self,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX] = None,
        theta: Union[np.ndarray, cs.SX] = None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(self._f):
            raise Exception('Model has no continuous-time dynamics!')
        if is_none(theta):
            theta = self.parameters.as_array()
        if is_none(w):
            w = ProcessNoise().as_array()
        if type(u) == np.ndarray:
            u = np.clip(u, self._lbu.as_array(), self._ubu.as_array())
        return self._f(x, u, w, theta)

    def forward_euler(
        self,
        f: Callable,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX],
        theta: Union[np.ndarray, cs.SX],
    ) -> Union[np.ndarray, cs.SX]:
        return x + dt * f(x, u, w, theta)

    def rk4(
        self,
        f: Callable,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX],
        theta: Union[np.ndarray, cs.SX],
    ) -> Union[np.ndarray, cs.SX]:
        k1 = f(x, u, w, theta)
        k2 = f(x + dt/2 * k1, u, w, theta)
        k3 = f(x + dt/2 * k2, u, w, theta)
        k4 = f(x + dt * k3, u, w, theta)
        x_next = x + dt/6 * (k1 +2*k2 +2*k3 + k4)
        x_next[3:7] = x_next[3:7] / cs.norm_2(x_next[3:7])
        return x_next


class NonlinearQuadrotorModel(DynamicsModel):
    def __init__(
        self,
        parameters: ModelParameters,
        lbu: Input = Input(get_config_values('lower_bound', INPUT_CONFIG)),
        ubu: Input = Input(get_config_values('upper_bound', INPUT_CONFIG)),
    ) -> None:
        super().__init__(
            parameters, lbu, ubu,
            STATE_CONFIG, INPUT_CONFIG, PROCESS_NOISE_CONFIG, 1
        )
        self._set_model()

    def _set_model(self) -> None:
        p = symbolic('position_wf', STATE_CONFIG)
        q = symbolic('attitude', STATE_CONFIG)
        vB = symbolic('linear_velocity_bf', STATE_CONFIG)
        wB = symbolic('angular_velocity_bf', STATE_CONFIG)
        x = cs.SX(cs.vertcat(p, q, vB, wB))

        m = symbolic('m', PARAMETER_CONFIG)
        a = symbolic('a', PARAMETER_CONFIG)
        Ixx = symbolic('Ixx', PARAMETER_CONFIG)
        Iyy = symbolic('Iyy', PARAMETER_CONFIG)
        Izz = symbolic('Izz', PARAMETER_CONFIG)
        b = symbolic('b', PARAMETER_CONFIG)
        r = symbolic('c', PARAMETER_CONFIG)
        s = symbolic('d', PARAMETER_CONFIG)
        theta = cs.SX(cs.vertcat(m, a, Ixx, Iyy, Izz, b, r, s))

        # Constants
        g = cs.SX(cs.vertcat(0, 0, -GRAVITY))
        A = cs.SX(cs.diag(a))
        J = cs.SX(cs.diag(cs.vertcat(Ixx, Iyy, Izz)))

        # Control input terms
        u = symbolic('normalized_squared_motor_speed', INPUT_CONFIG)
        K = cs.SX(cs.vertcat(
            cs.SX.zeros(2, self.nu),
            cs.SX.ones(1, self.nu),
        ))
        B = cs.SX(cs.vertcat(
            s.T,
            -r.T,
            (b * alternating_ones(self.nu)).T,
        ))

        # Additive process noise
        w = cs.SX.sym('w', self.nw)

        # Continuous-time dynamics
        xdot = w + cs.SX(cs.vertcat(
            quat.Q(q) @ vB,
            0.5 * quat.G(q) @ wB,
            quat.Q(q).T @ g + (K @ u - A @ vB) / m - cs.cross(wB, vB),
            cs.inv(J) @ (B @ u - cs.cross(wB, J @ wB))
        ))

        # Define dynamics function
        self._f = cs.Function(
            'f_NonlinearQuadrotorModel',
            [x, u, w, theta], [xdot]
        )


class KoopmanLiftedQuadrotorModel(DynamicsModel):
    def __init__(
        self,
        observables_order: int,
        parameters: ModelParameters,
        lbu: Input = get_config_values('lower_bound', INPUT_CONFIG),
        ubu: Input = get_config_values('upper_bound', INPUT_CONFIG),
    ) -> None:
        super().__init__(
            parameters, lbu, ubu, KOOPMAN_STATE_CONFIG, INPUT_CONFIG,
            KOOPMAN_PROCESS_NOISE_CONFIG, observables_order
        )
        self._set_lifted_model()

    def F(
        self,
        dt: float,
        z0: Union[np.ndarray, cs.SX],
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX] = None,
        theta: Union[np.ndarray, cs.SX] = None,
    ) -> Union[np.ndarray, cs.SX]:
        xf = self.rk4(self.f, dt, z0, x, u, w, theta)
        if type(xf) == cs.DM:
            return np.array(xf).flatten()
        else:
            return xf

    def f(
        self,
        z0: Union[np.ndarray, cs.SX],
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX] = None,
        theta: Union[np.ndarray, cs.SX] = None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(self._f):
            raise Exception('Model has no continuous-time dynamics!')
        if is_none(theta):
            theta = self.parameters.as_array()
        if is_none(w):
            #w = ProcessNoise().as_array()
            w = KoopmanLiftedProcessNoise(order=self.order).as_array()
        return self._f(z0, x, u, w, theta)

    def rk4(
        self,
        f: Callable,
        dt: float,
        z0: Union[np.ndarray, cs.SX],
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX],
        theta: Union[np.ndarray, cs.SX],
    ) -> Union[np.ndarray, cs.SX]:
        k1 = f(z0, x, u, w, theta)
        k2 = f(z0, x + dt/2 * k1, u, w, theta)
        k3 = f(z0, x + dt/2 * k2, u, w, theta)
        k4 = f(z0, x + dt * k3, u, w, theta)
        return x + dt/6 * (k1 +2*k2 +2*k3 +k4)

    def _set_lifted_model(self) -> None:
        pB_0 = symbolic('position_bf', KOOPMAN_STATE_CONFIG)
        vB_0 = symbolic('linear_velocity_bf', KOOPMAN_STATE_CONFIG)
        gB_0 = symbolic('gravity_bf', KOOPMAN_STATE_CONFIG)
        wB_0 = symbolic('angular_velocity_bf', KOOPMAN_STATE_CONFIG)
        z0 = cs.vertcat(pB_0, vB_0, gB_0, wB_0)

        pB = symbolic('position_bf', KOOPMAN_STATE_CONFIG, self._order)
        vB = symbolic('linear_velocity_bf', KOOPMAN_STATE_CONFIG, self._order)
        gB = symbolic('gravity_bf', KOOPMAN_STATE_CONFIG, self._order)
        wB = symbolic('angular_velocity_bf', KOOPMAN_STATE_CONFIG, self._order)
        z = cs.vertcat(pB, vB, gB, wB)

        m = symbolic('m', PARAMETER_CONFIG)
        a = symbolic('a', PARAMETER_CONFIG)
        Ixx = symbolic('Ixx', PARAMETER_CONFIG)
        Iyy = symbolic('Iyy', PARAMETER_CONFIG)
        Izz = symbolic('Izz', PARAMETER_CONFIG)
        b = symbolic('b', PARAMETER_CONFIG)
        r = symbolic('c', PARAMETER_CONFIG)
        s = symbolic('d', PARAMETER_CONFIG)
        theta = cs.SX(cs.vertcat(m, a, Ixx, Iyy, Izz, b, r, s))

        J = cs.SX(cs.diag(cs.vertcat(Ixx, Iyy, Izz)))

        # Derive Koopman observables
        ws_N = attitude.get_ws(wB_0, J, self._order)
        Hs_N = attitude.get_Hs(ws_N, J)

        ps = position.get_ps(pB_0, ws_N)
        Ps = position.get_Ps(ps, ws_N, Hs_N, J)

        vs = velocity.get_vs(vB_0, ws_N)
        Os = velocity.get_Os(vs)
        Vs = velocity.get_Vs(vs, ws_N, Hs_N, J)

        gs = gravity.get_gs(gB_0, ws_N)
        Gs = gravity.get_Gs(gs, ws_N, Hs_N, J)

        ws = attitude.get_ws(wB_0, J, self._order)
        Hs = attitude.get_Hs(ws, J)

        # Control input terms
        K = cs.SX(cs.vertcat(
            cs.SX.zeros(2, self.nu),
            cs.SX.ones(1, self.nu),
        ))
        B = cs.SX(cs.vertcat(
            s.T,
            -r.T,
            (b * alternating_ones(self.nu)).T,
        ))
        u = symbolic('normalized_squared_motor_speed', INPUT_CONFIG)

        # Process noise
        w = cs.SX.sym('w', self.nw)

        # Continuous-time dynamics
        B_ode = cs.SX(
            get_input_matrix(Ps, Os, Vs, Gs, Hs, J, m) @ cs.vertcat(K, B))
        A_ode = cs.SX( get_state_matrix(self._order, self._order) )
        zdot = A_ode @ z + B_ode @ u + w

        # Define dynamics function
        self._f = cs.Function(
            'f_KoopmanLiftedQuadrotorModel',
            [z0, z, u, w, theta], [zdot]
        )


#TODO: add more models
def CrazyflieModel(a=np.zeros(3)) -> NonlinearQuadrotorModel:
    '''
    Crazyflie system identification: https://arxiv.org/pdf/1608.05786
    '''
    params = ModelParameters()
    params.set_member('m', 0.027)
    params.set_member('a', a)
    params.set_member('Ixx', 1.436 * 10**-5)
    params.set_member('Iyy', 1.395 * 10**-5)
    params.set_member('Izz', 2.173 * 10**-5)
    params.set_member('c', 0.0283 * np.array([1.0, 1.0, -1.0, -1.0]))
    params.set_member('d', 0.0283 * np.array([1.0, -1.0, -1.0, 1.0]))
    k = 3.1582e-10
    b = 7.9379e-12
    params.set_member('b', b/k * np.ones(4))

    pwm_to_rpm = lambda pwm: 0.2685 * pwm + 4070.3
    pwm_max = 65535
    lbu = Input(np.zeros(4))
    ubu = Input(k * pwm_to_rpm(pwm_max)**2 * np.ones(4))
    return NonlinearQuadrotorModel(params, lbu, ubu)


def FusionOneModel(a=np.zeros(3)) -> NonlinearQuadrotorModel:
    '''
    Fusion 1 quadrotor identification: https://arc.aiaa.org/doi/epdf/10.2514/6.2020-1238
    '''
    params = ModelParameters()
    params.set_member('m', 0.250)
    params.set_member('a', a)
    params.set_member('Ixx', 4.27e-4)
    params.set_member('Iyy', 6.09e-4)
    params.set_member('Izz', 1.50e-3)
    params.set_member('c', 0.0635 * np.array([1.0, 1.0, -1.0, -1.0]))
    params.set_member('d', 0.0635 * np.array([1.0, -1.0, -1.0, 1.0]))
    DR = 0.0584
    CT = 0.279
    CP = 0.333
    b = CP * DR / (2 * np.pi * CT)
    params.set_member('b', b * np.ones(4))

    lbu = Input(np.zeros(4))
    rho = 1.204
    k = rho * CT * DR**4 / (2 * np.pi)**2
    wbu = 35 * 1000 * 2 * np.pi / 60
    ubu = Input(k * wbu**2 * np.ones(4))
    return NonlinearQuadrotorModel(params, lbu, ubu)
