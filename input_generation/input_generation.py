import numpy as np
from typing import Union
from input_generation.pde_generation_data import OdeFn
from abc import abstractmethod
import tensorflow as tf


class DataGeneration:

    def __init__(self, solver: OdeFn, initial_state_number: int, time_duration: int, dt: float, t_init: float,
                 rng: Union[np.random.RandomState, int] = None, add_white_gaussian: tuple = None):
        """

        :param initial_state_number: number of initial states to generate
        :param time_step_number: number of time step
        :param rng: random framework
        :param add_white_gaussian: tuple(mean, standard deviation), if None, no white gaussian noise
        """
        self._solver = solver
        self.t_init = t_init
        eps = np.finfo(np.float32).eps
        self.t = np.arange(t_init, t_init + time_duration - eps, dt).astype(np.float32)
        if isinstance(rng, np.random.RandomState):
            self._rng = rng
        elif isinstance(rng, int):
            self._rng = np.random.RandomState()
            self._rng.seed(rng)
        else:
            self._rng = np.random

        self._init_states_nb = initial_state_number
        self._ts_nb = self.t.shape[0]
        self._res = None

        self._white_gaussian = False
        self._wg_noise_mean = 0.
        self._wg_noise_std = 0.
        if add_white_gaussian is not None and isinstance(add_white_gaussian, (tuple, list)):
            self._white_gaussian = True
            self._wg_noise_mean = add_white_gaussian[0]
            self._wg_noise_std = add_white_gaussian[1]
        self._res = None

    @abstractmethod
    def generate_initial_states(self, **kwargs):
        pass

    def add_white_gaussian_noise(self, mean=0, std=1):
        return self._rng.randn(self._ts_nb) * std + mean

    def get_one_case_idx(self):
        return self._rng.randint(self._init_states_nb)

    def add_one_case(self, initial_condition: Union[tuple, list]):
        """

        :param initial_condition: initial condition (if there are an initial position and an initial speed,
        two initial values in one tuple or list)
        :return:
        """
        white_gaussian_noise = np.zeros(self.t.shape[0])
        if self._white_gaussian:
            white_gaussian_noise = self.add_white_gaussian_noise(mean=self._wg_noise_mean, std=self._wg_noise_std)
        res = self._res
        new_res = np.zeros((self._ts_nb, 1, len(initial_condition)))
        res = np.concatenate([new_res]) if res is None else np.concatenate([res, new_res], axis=1)
        x_init = tf.constant(initial_condition, dtype=tf.float32)
        results = self._solver.solve(self.t_init, x_init, solution_times=self.t)
        res[:, -1, 0] = results.states.numpy()[:, 0] + white_gaussian_noise
        res[:, -1, 1] = results.states.numpy()[:, 1] + white_gaussian_noise
        self._res = res
        self._init_states_nb += 1
        return res[:, -1, :]

