import pandas as pd
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import phoenix_drone_simulation  # noqa
import time
from scipy import signal


class LowPassFilter:
    def __init__(self, gain: float, time_constant: float, sample_time: float):
        """

        Parameters
        ----------
        gain: K
        time_constant: T
            The filter time constant. Remember: ~4T until step response
        sample_time: T_s
            Typically 1/sim_freq
        """
        self._x = 0
        self.K = gain
        self.T = time_constant
        self.T_s = sample_time

    def apply(self, u):
        ratio = self.T_s / self.T
        self._x = (1 - ratio) * self._x + self.K * ratio * u
        return self._x

    def set(self, x):
        self._x = x


def main():
    SIM_FREQ = 1000
    gyro_lpf = LowPassFilter(gain=1., time_constant=2/SIM_FREQ, sample_time=1/SIM_FREQ)

    ts = np.arange(500) / SIM_FREQ
    signal = np.sin(ts*6) + np.random.normal(0, 0.03, size=ts.shape) \
             + np.random.uniform(-0.03, 0.03, size=ts.shape)

    gyro_lpf.set(signal[0])

    filtered_sig = signal.copy()

    for i in range(1, len(signal)):
        filtered_sig[i] = gyro_lpf.apply(signal[i])

    plt.plot(ts, signal)
    plt.plot(ts, filtered_sig)
    plt.show()


if __name__ == '__main__':
    main()

