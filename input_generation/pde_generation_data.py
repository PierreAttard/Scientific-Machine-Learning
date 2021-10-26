from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Union
import numpy as np


class OdeFn:
    """
    Dormand–Prince method to solve ordinary differential equations **Dormand and Prince 1980**
    """

    def __init__(self, u: np.array):
        """
        u is the second member of the differential equation
        :param u:
        """
        self._u = u

    def __call__(self, t, x):
        return self.ode_fn(t, x, self._u)

    @property
    def u(self) -> None:
        return self._u

    @u.setter
    def u(self, new_u: np.array) -> None:
        self._u = new_u

    @abstractmethod
    def ode_fn(self, t, x, u):
        pass

    def solve(self, t_init: Union[float, tf.Tensor], x_init: Union[float, tf.Tensor], solution_times: np.array):
        rf5 = tfp.math.ode.DormandPrince()
        results = rf5.solve(self, t_init, x_init, solution_times=solution_times)
        return results

    # def ode_fn(self, t, x, u):
    #     return tf.Variable([x[1], u[int(t.numpy()) ] -(omeg a* *2) * np.sin(x[0]) - alpha * x[1]])