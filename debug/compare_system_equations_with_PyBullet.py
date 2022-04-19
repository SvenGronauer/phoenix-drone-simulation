import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pb

# local imports
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv, DroneHoverSimpleEnv
from phoenix_drone_simulation.envs.control import Attitude


if __name__ == '__main__':
    simle_env = DroneHoverSimpleEnv()
    bullet_env = DroneHoverBulletEnv()


    # disable domain randomization and observation noise
    bullet_env.unwrapped.domain_randomization = -1
    simle_env.unwrapped.domain_randomization = -1
    bullet_env.unwrapped.observation_noise = -1
    simle_env.unwrapped.observation_noise = -1
    bullet_env.enable_reset_distribution = False
    simle_env.enable_reset_distribution = False

    # disable motor dynamics
    bullet_env.drone.USE_LATENCY = False
    simle_env.drone.USE_LATENCY = False
    bullet_env.drone.use_latency = False
    simle_env.drone.use_motor_dynamics = False

    time_step = 0.01
    N = 50
    obs_0 = bullet_env.reset()
    obs_1 = simle_env.reset()
    xs = np.zeros((N, 2))
    ys = np.zeros((N, 2))
    zs = np.zeros((N, 2))
    rolls = np.zeros((N, 2))
    pitchs = np.zeros((N, 2))
    yaws = np.zeros((N, 2))

    rpy_dots = np.zeros((N, 2, 3))
    quats = np.zeros((N, 2, 4))

    for i in range(N):
        # time.sleep(0.5)
        action = 0.01 * np.ones(4)
        action[3] = 0.5

        #
        xs[i] = [bullet_env.drone.xyz[0], simle_env.drone.xyz[0]]  # track x position
        ys[i] = [bullet_env.drone.xyz[1], simle_env.drone.xyz[1]]  # track y position
        zs[i] = [bullet_env.drone.xyz[2], simle_env.drone.xyz[2]]  # track z position
        rolls[i] = [bullet_env.drone.rpy[0], simle_env.drone.rpy[0]]
        pitchs[i] = [bullet_env.drone.rpy[1], simle_env.drone.rpy[1]]

        rpy_dots[i, 0, :] = bullet_env.drone.rpy_dot
        rpy_dots[i, 1, :] = simle_env.drone.rpy_dot
        quats[i, 0, :] = bullet_env.drone.quaternion
        quats[i, 1, :] = simle_env.drone.quaternion
        
        bullet_env.step(action)
        simle_env.step(action)


    bullet_env.close()
    fig = plt.figure()
    ts = np.arange(N) * time_step

    print(rpy_dots[:, 1, 0])
    print(rpy_dots[:, 1, 0].shape)



    ax = fig.add_subplot(441)
    ax.title.set_text('Roll')
    plt.plot(ts, rolls[:, 0], label='Roll (Bullet)', color='red')
    plt.plot(ts, rolls[:, 1], label='Roll (Simple)', color='blue')
    plt.legend()

    ax = fig.add_subplot(442)
    ax.title.set_text('Pitch')
    plt.plot(ts, pitchs[:, 0], label='Pitch (Bullet)', color='red')
    plt.plot(ts, pitchs[:, 1], label='Pitch (Simple)', color='blue')
    plt.legend()

    ax1 = fig.add_subplot(443)
    ax1.title.set_text('Z Position')
    plt.plot(ts, zs[:, 0], label='z (Bullet)')
    plt.plot(ts, zs[:, 1], label='z (Simple)')
    plt.legend()

    # ======> Attitude Rate <========

    ax = fig.add_subplot(445)
    ax.title.set_text('Roll_dot')
    plt.plot(ts, rpy_dots[:, 0, 0], label='Bullet', color='red')
    plt.plot(ts, rpy_dots[:, 1, 0], label='Simple', color='blue')
    plt.legend()

    ax = fig.add_subplot(446)
    ax.title.set_text('Pitch_dot')
    plt.plot(ts, rpy_dots[:, 0, 1], label='Bullet', color='red')
    plt.plot(ts, rpy_dots[:, 1, 1], label='Simple', color='blue')
    plt.legend()

    ax = fig.add_subplot(447)
    ax.title.set_text('Yaw_dot')
    plt.plot(ts, rpy_dots[:, 0, 2], label='Bullet', color='red')
    plt.plot(ts, rpy_dots[:, 1, 2], label='Simple', color='blue')
    plt.legend()

    # ======> Quaternions <========

    ax = fig.add_subplot(4,4,9)
    ax.title.set_text('Quaternion[0]')
    plt.plot(ts, quats[:, 0, 0], label='Bullet', color='red')
    plt.plot(ts, quats[:, 1, 0], label='Simple', color='blue')
    plt.legend()

    ax = fig.add_subplot(4,4,10)
    ax.title.set_text('Quaternion[1]')
    plt.plot(ts, quats[:, 0, 1], label='Bullet', color='red')
    plt.plot(ts, quats[:, 1, 1], label='Simple', color='blue')
    plt.legend()

    ax = fig.add_subplot(4,4,11)
    ax.title.set_text('Quaternion[2]')
    plt.plot(ts, quats[:, 0, 2], label='Bullet', color='red')
    plt.plot(ts, quats[:, 1, 2], label='Simple', color='blue')
    plt.legend()

    ax = fig.add_subplot(4,4,12)
    ax.title.set_text('Quaternion[3]')
    plt.plot(ts, quats[:, 0, 3], label='Bullet', color='red')
    plt.plot(ts, quats[:, 1, 3], label='Simple', color='blue')
    plt.legend()

    plt.show()
