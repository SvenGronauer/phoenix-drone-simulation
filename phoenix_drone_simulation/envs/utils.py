r""""Environment utilities for Project Phoenix.

Author:     Sven Gronauer
Created:    17.05.2021

Updates:
    14.07.2021: Added low-pass filter and sensor noise (according to USC)
    05.08.2021: Added transformation function for quaternions, rotation matrices
    14.04.2022: Add depreciation warnings for future clean-ups
"""
import os
import numpy as np
from math import exp


def rad2deg(x):
    r"""Converts radians to degrees."""
    return 180*x/np.pi


def deg2rad(x):
    r"""Converts degrees to radians."""
    return np.pi*x/180


def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'assets')
    return data_path


def get_quaternion_from_euler(rpy) -> np.ndarray:
    r"""Get quaternion [x, y, z, w] from Euler angles RPY [rad].

    See:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    assert rpy.shape == (3, )
    roll, pitch, yaw = rpy
    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = np.cos(halfYaw)
    sinYaw = np.sin(halfYaw)
    cosPitch = np.cos(halfPitch)
    sinPitch = np.sin(halfPitch)
    cosRoll = np.cos(halfRoll)
    sinRoll = np.sin(halfRoll)

    x = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
    y = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
    z = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw
    w = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw

    # Note: PyBullet uses the order: [X,Y,Z,W]
    return np.array([x, y, z, w])


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


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
