import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import seaborn as sns

# global line, pt


class DrawFrame:

    def __init__(self, results, L, t):
        self._results = results
        self.L = L
        self.t = t

    def init_graph(self):
        fig = plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 1, 1)

        # set up the subplots as needed
        L = self.L
        ax.set_xlim((-L, L))
        ax.set_ylim((-L, 0))
        # ax.set_xlim((-12, 12))
        # ax.set_ylim((-12, 12))
        ax.set_xlabel('Time')
        ax.set_ylabel('')
        ax.set_title('Phase Plane')

        # create objects that will change in the animation. These are
        # initially empty, and will be given new values for each frame
        # in the animation.
        txt_title = ax.set_title('')

        self.pt, = ax.plot([], [], 'g.', ms=20)
        self.line, = ax.plot([], [], 'y', lw=2)
        return fig, ax, self.pt, self.line

    def init_graph_multi(self, legend=None):
        fig = plt.figure(figsize=(12, 5))
        L = self.L
        ax = plt.subplot(1, 1, 1)
        # set up the subplots as needed
        ax.set_xlim((-L, L))
        ax.set_ylim((-L, 0))
        # ax.set_xlim((-12, 12))
        # ax.set_ylim((-12, 12))
        ax.set_xlabel('Time')
        ax.set_ylabel('')

        n_pendulum = self._results.shape[1]
        colors = sns.color_palette("husl", n_pendulum)
        self.lines = sum([ax.plot([], [], color=colors[k], lw=2) for k in range(n_pendulum)], [])
        if legend is not None:
            ax.legend(legend)
        self.pts = sum([ax.plot([], [], 'o', color=colors[k], ms=15) for k in range(n_pendulum)], [])


        return fig

    def drawframe(self, t):
        theta = self._results[t] - np.pi
        # space_step = np.linspace(0, self.L+1, 100)
        space_step = 100
        x = np.linspace(0, self.L * np.sin(theta), space_step)
        y = np.linspace(0, self.L * np.cos(theta), space_step)
        self.line.set_data(x, y)
        self.pt.set_data(x[-1], y[-1])
        return self.line,

    def drawframe_multi(self, t):
        n_pendulum = self._results.shape[1]
        thetas = self._results[t, :] - np.pi
        # space_step = np.linspace(0, self.L+1, 100)
        space_step = 100
        xs = np.linspace(0, self.L * np.sin(thetas), space_step)
        ys = np.linspace(0, self.L * np.cos(thetas), space_step)

        for i in range(n_pendulum):
            self.lines[i].set_data(xs[:, i], ys[:, i])
            self.pts[i].set_data(xs[-1, i], ys[-1, i])
        return tuple(self.lines)

    def get_anim(self, fig, interval=20, blit=True, multi=False):
        if interval is None:
            t = self.t
            dt = t[1] - t[0]
            interval = dt * 1000  # seconds to miliseconds

        func = self.drawframe
        if multi:
            func = self.drawframe_multi
        return animation.FuncAnimation(fig, func, frames=self._results.shape[0], interval=interval, blit=blit)

    def plot_theta(self, figsize=(12, 5)):
        eps = 0.1
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        ax.plot(self.t, self._results)
        ax.set_ylim((-np.pi - eps, np.pi + eps))
        ax.set_xlabel('Time')
        ax.set_ylabel('Theta')
        plt.show()
        return