r"""Plot utilities for simulation optimization.

Author:     Sven Gronauer (sven.gronauer@gmail.com)
Created:    20-08-2021
"""
import pandas as pd
import numpy as np
import torch
import gym
import pybullet as pb
import matplotlib.pyplot as plt
import phoenix_drone_simulation  # noqa
from typing import Optional
import phoenix_drone_simulation.simopt.config as so_conf


def plot_trajectories(
        obs_real: np.ndarray,
        obs_sim: np.ndarray,
        actions: Optional[np.ndarray]
):
    r"""Plot trajectories of real and simulated data."""

    ts = np.arange(obs_real.shape[0])
    M, N = 4, 4
    fig, axes = plt.subplots(M, N, figsize=(8, 6))
    fig.tight_layout()
    ax_num = 0

    """"==== XYZ ===="""
    x_real = obs_real[:, so_conf.xyz_real_slice]
    x_sim = obs_sim[:, so_conf.xyz_sim_slice]
    for i, col in enumerate(['x', 'y', 'z']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        ax.plot(ts, x_real[:, i], label=col.upper()+' (real)')
        ax.plot(ts, x_sim[:, i], label=col.upper()+' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1

    """"==== Linear Velocities ===="""
    ax_num = N
    x_real = obs_real[:, so_conf.xyz_dot_real_slice]
    x_sim = obs_sim[:, so_conf.xyz_dot_sim_slice]
    for i, col in enumerate(['x_dot', 'y_dot', 'z_dot']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col + ' [m/s]')
        ax.plot(ts, x_real[:, i], label=col.upper()+' (real)')
        ax.plot(ts, x_sim[:, i], label=col.upper()+' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1

    """"==== RPY ===="""

    ax_num = 2 * N
    # transform quaternions -> RPY
    rpy_real = obs_real[:, so_conf.rpy_real_slice]

    quat_sim = obs_sim[:, so_conf.quaternion_sim_slice]
    rpy_sim = np.empty((obs_real.shape[0], 3))
    for i, row in enumerate(quat_sim):  # iterate over quaternions
        rpy_sim[i] = np.asarray(pb.getEulerFromQuaternion(row))

    for i, col in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        ax.plot(ts, rpy_real[:, i] * 180 / np.pi, label=col.upper()+' (real)')
        ax.plot(ts, rpy_sim[:, i] * 180 / np.pi, label=col.upper()+' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1

    """"==== Attitude Rates ===="""
    ax_num = 3 * N
    rpy_dot_real = obs_real[:, so_conf.rpy_dot_real_slice] * 180 / np.pi
    rpy_dot_sim = obs_sim[:, so_conf.rpy_dot_sim_slice] * 180 / np.pi

    for i, col1 in enumerate(['roll_dot', 'pitch_dot', 'yaw_dot']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col1 + ' [deg/s]')
        ax.plot(ts, rpy_dot_real[:, i], label=col1.upper() + ' (real)')
        ax.plot(ts, rpy_dot_sim[:, i], label=col1.upper() + ' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1
    #
    """"==== PWMs ===="""
    ax_num = M - 1
    if actions is not None:
        # actions = get_nn_outputs(df, pi, obs_rms=obs_rms)
        for i, col in enumerate(['mot0', 'mot1', 'mot2', 'mot3']):
            ax = axes.flatten()[ax_num]
            ax.set_title(col)
            # ax.plot(ts, df[col], label=col.upper() + ' (real)')
            # motor_label = f'mot{i}'
            # ax.plot(ts, df[motor_label], label=motor_label.upper()+' (real)')
            ax.plot(ts, actions[:, i], label=col.upper())
            # ax.set_ylim(0, 2**16)
            ax.legend()
            ax.grid()
            ax_num += N

    plt.show()
