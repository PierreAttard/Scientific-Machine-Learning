from input_generation.pde_generation_data import OdeFn
from input_generation.input_generation import DataGeneration
import numpy as np
import tensorflow as tf
from typing import Union


class OdePendulum(OdeFn):
    """
    Partial equation differential to generate data with

    d2Theta/dt2 - omega^2.Theta - alpha.dTheta/dt = u(t)

    """

    def __init__(self, u, omega: float, alpha: float):
        super(OdePendulum, self).__init__(u)
        self._omega = omega
        self._alpha = alpha

    def ode_fn(self, t, x, u):
        return tf.Variable([x[1], u[int(t.numpy())] - (self._omega ** 2) * np.sin(x[0]) - self._alpha * x[1]])


class DampedPendulumDataGeneration(DataGeneration):

    def __init__(self, solver: OdeFn, initial_state_number: int, time_duration: int, dt: float, t_init: float,
                 rng: Union[np.random.RandomState, int] = None, add_white_gaussian: tuple = None):
        """

        :param initial_state_number: number of initial states to generate
        :param time_duration: number of time step
        :param rng: random framework
        :param add_white_gaussian: tuple(mean, standard deviation), if None, no white gaussian noise
        """
        super(DampedPendulumDataGeneration, self).__init__(solver, initial_state_number, time_duration, dt, t_init, rng,
                                                           add_white_gaussian)

    def generate_initial_states(self, angle_bounds: Union[tuple, list], speed_bounds: Union[tuple, list]):
        """
        generate a set of initial angle ande initial angle speed
        :param angle_bounds:
        :param speed_bounds:
        :return:
        """
        angle_lb, angle_up = angle_bounds
        speed_lb, speed_up = speed_bounds
        angle_inits = np.linspace(angle_lb, angle_up, self._init_states_nb)
        speed_inits = np.linspace(speed_lb, speed_up, self._init_states_nb)
        init_states_idx = [(self._rng.randint(self._init_states_nb), self._rng.randint(self._init_states_nb))
                           for _ in range(self._ts_nb)]
        init_states = np.array([(angle_inits[i], speed_inits[j]) for i, j in init_states_idx])
        return init_states

    def generate_sample(self, angle_bounds: Union[tuple, list],
                        speed_bounds: Union[tuple, list]):
        """
        generate a set of initial angle ande initial angle speed
        :param angle_bounds:
        :param speed_bounds:
        :return:
        """
        init_states = self.generate_initial_states(angle_bounds, speed_bounds)
        res = np.zeros((self.t.shape[0], init_states.shape[0], init_states.shape[1]))
        white_gaussian_noise = np.zeros(self.t.shape[0])
        if self._white_gaussian:
            white_gaussian_noise = self.add_white_gaussian_noise(mean=self._wg_noise_mean, std=self._wg_noise_std)
        for i, (theta0, v0) in enumerate(init_states):
            x_init = tf.constant([theta0, v0], dtype=tf.float64)
            results = self._solver.solve(self.t_init, x_init, solution_times=self.t)
            res[:, i, 0] = results.states.numpy()[:, 0] + white_gaussian_noise
            res[:, i, 1] = results.states.numpy()[:, 1] + white_gaussian_noise
        self._res = res
        return res

    def get_one_sample(self, idx=None):
        if idx is None:
            idx = self.get_one_case_idx()
        return self._res[:, idx, :]
