import numpy as np
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
from tensorflow import float32, concat, convert_to_tensor, linalg, math
import tensorflow as tf


class RungeKuttaIntegratorCell(Layer):
    def __init__(self, omega, alpha, dt, initial_state, **kwargs):
        super(RungeKuttaIntegratorCell, self).__init__(**kwargs)
        self._omega = convert_to_tensor(omega)
        self._alpha = convert_to_tensor(alpha)
        self.initial_state = initial_state
        self.state_size = 2
        self.A = np.array([0., 0.5, 0.5, 1.0], dtype='float32')
        self.B = np.array([[1 / 6, 2 / 6, 2 / 6, 1 / 6]], dtype='float32')
        self.dt = dt

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("ALPHA", shape=self._alpha.shape, trainable=True,
                                      initializer=lambda shape, dtype: self._alpha, **kwargs)
        self.built = True

    def call(self, inputs, states):
        """
        Runge–Kutta methods order 4 application. Classic RK
        :param inputs: pendulum angle
        :param states: pendulum angle and speed
        :return: new_inputs, new_states
        """
        alpha = self.kernel
        theta = states[0][:, :1]
        theta_dot = states[0][:, 1:]

        theta_ddoti = self._fun(self._omega, alpha, inputs, theta, theta_dot)
        thetai = theta + self.A[0] * theta_dot * self.dt
        theta_doti = theta_dot + self.A[0] * theta_ddoti * self.dt
        fn = self._fun(self._omega, alpha, inputs, thetai, theta_doti)
        for j in range(1, 4):
            thetan = theta + self.A[j] * theta_dot * self.dt
            theta_dotn = theta_dot + self.A[j] * theta_ddoti * self.dt
            theta_doti = concat([theta_doti, theta_dotn], axis=0)
            fn = concat([fn, self._fun(self._omega, alpha, inputs, thetan, theta_dotn)], axis=0)

        theta = theta + linalg.matmul(self.B, theta_doti) * self.dt
        theta_dot = theta_dot + linalg.matmul(self.B, fn) * self.dt
        return theta, [concat(([theta, theta_dot]), axis=-1)]

    def _fun(self, omega, alpha, inputs, theta, theta_dot):
        """

        :param omega: proper pulsation
        :param alpha: damped parameter
        :param inputs: external stimulus
        :param theta: pendulum angle
        :param theta_dot: pendulum angle speed
        :return:
        """
        return inputs - math.pow(omega, tf.constant([2], dtype="float32")) * math.sin(theta) - alpha * theta_dot

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.initial_state


def create_model(omega, alpha, dt, initial_state, ub, lb, batch_input_shape, return_sequences=True, unroll=False,
                 optimizer=None):
    rkCell = RungeKuttaIntegratorCell(omega=omega, alpha=alpha, dt=dt, initial_state=initial_state)
    PINN = RNN(cell=rkCell, batch_input_shape=batch_input_shape, return_sequences=return_sequences, return_state=False,
               unroll=unroll)
    model = Sequential()
    if all(map(lambda x: x is not None and isinstance(x, (int, float)) and x != 0, [ub, lb])):
        model.add(tf.keras.layers.Lambda(lambda X: 2*(X - lb)/(ub - lb) - 1))
    model.add(PINN)
    if optimizer is None:
        optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    # model.compile(loss='mse', optimizer=RMSprop(1e4), metrics=['mae'])
    # model.compile(loss='mse', optimizer="adams", metrics=['mae'])
    return model