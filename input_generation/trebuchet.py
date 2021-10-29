import numpy as np
import matplotlib.pyplot as plt


class Trebuchet:

    def __init__(self, initial_angle, initial_speed, initial_y, mass, damped, dt):
        """

        :param initial_angle:
        :param mass:
        :param damped:
        """
        self._init_angle = initial_angle
        self._init_speed = initial_speed
        self._init_y = initial_y
        self._m = mass
        self._damped = damped
        self._g = 9.82
        self._dt = dt
        self._compute_params()

    def _compute_params(self):
        self._initial_speed_x = np.cos(self._init_angle) * self._init_speed
        self._initial_speed_y = np.sin(self._init_angle) * self._init_speed
        self._alpha = self._damped

    def x(self, t):
        am = self._alpha / self._m
        return self._initial_speed_x * (1 / am) * (1 - np.exp(-am * t))

    def dx_dt(self, t):
        am = self._alpha / self._m
        return - self._initial_speed_x * (1 - np.exp(-am * t))

    def y(self, t):
        am = self._alpha / self._m
        return (1 / am) * (self._initial_speed_y + self._g / am) * (1 - np.exp(-am * t)) - (self._g / am) * t + \
               self._init_y

    def dy_dt(self, t):
        am = self._alpha / self._m
        print(- (self._initial_speed_y + self._g / am) * (1 - np.exp(-am * t)) - (self._g / am))
        print((1 - np.exp(-am * t)))
        return - (self._initial_speed_y + self._g / am) * (1 - np.exp(-am * t)) - (self._g / am)

    def get_tf(self):
        t_f = self._initial_speed_y / self._g
        y = self.y(t_f)
        dt = 1
        k = 0.9
        while y < 0 or y > 1e-4:
            if np.abs(y / dt) < 5:
                dt = dt * k
            if 1e-4 > y > 0:
                break
            if y > 0:
                t_f = t_f + dt
                y = self.y(t_f)
            else:
                t_f = t_f - dt
                if t_f <0:
                    t_f = -t_f
                y = self.y(t_f)
            print(f"y={y:.2f}, t_f={t_f:.2f}, dt={dt:.5f}")
        return t_f


class Trebuchet2:

    def __init__(self, initial_angle, initial_speed, initial_y, mass, wind: tuple, dt):
        """

        :param initial_angle:
        :param masse:
        :param wind:
        """
        self._init_angle = initial_angle
        self._init_speed = initial_speed
        self._init_y = initial_y
        self._m = mass
        self._wind_x, self._wind_y = wind
        self._g = 9.82
        self._dt = dt
        self._compute_params()

    def _compute_params(self):
        self._initial_speed_x = np.cos(self._init_angle) * self._init_speed
        self._initial_speed_y = np.sin(self._init_angle) * self._init_speed

    def x(self, t):
        return 0.5 * (self._wind_x / self._m) * np.power(t, 2) + self._initial_speed_x * t

    def dx_dt(self, t):
        return (self._wind_x / self._m) * t + self._initial_speed_x

    def y(self, t):
        return 0.5 * ((self._wind_y / self._m) - self._g) * np.power(t, 2) + self._initial_speed_y * t + self._init_y

    def dy_dt(self, t):
        return ((self._wind_y / self._m) - self._g) * t + self._initial_speed_y

    def get_tf(self):
        t_f = self._initial_speed_y / self._g
        y = self.y(t_f)
        dt = 1
        k = 0.9
        while y < 0 or y > 1e-4:
            if np.abs(y / dt) < 5:
                dt = dt * k
            if 1e-4 > y > 0:
                break
            if y > 0:
                t_f = t_f + dt
                y = self.y(t_f)
            else:
                t_f = t_f - dt
                if t_f <0:
                    t_f = -t_f
                y = self.y(t_f)
            print(f"y={y:.2f}, t_f={t_f:.2f}, dt={dt:.5f}")
        return t_f


if __name__ == "__main__":

    treb = Trebuchet2(initial_angle=np.pi / 4, initial_speed=3, initial_y=0.1, mass=1, wind=(-3, 0), dt=1)
    t_f = treb.get_tf()
    print(t_f)
    L = treb.x(t_f)
    print(f"L = {L:.2f}m")
    ts = np.linspace(0, t_f, 100)
    y = treb.y(ts)
    x = treb.x(ts)
    plt.plot(ts, y)
    plt.show()

    plt.plot(x, y)
    plt.show()

    if False:
        treb = Trebuchet(initial_angle=np.pi / 4, initial_speed=1, initial_y=0, mass=1, damped=1, dt=1)

        ts = np.linspace(0, 0.2, 100)
        ydot = treb.dy_dt(ts)
        plt.plot(ts, ydot)
        plt.show()

        y = treb.y(ts)
        x = treb.x(ts)
        plt.plot(ts, y)
        plt.show()
        plt.plot(x, y)
        plt.show()
        print(treb.get_tf())