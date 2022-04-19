"""Use an Attitude Controller to stabilize the drone.

Note: the thrust of the drone is hard-coded.

Author:     Sven Gronauer
Created:    24.09.2021
"""
import time
import numpy as np

# local imports
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv, DroneHoverSimpleEnv
from phoenix_drone_simulation.envs.control import Attitude


def main():
    env = DroneHoverBulletEnv()
    # Important: call render before updating the controller and domain rand
    env.render()

    # set PID
    env.drone.control = Attitude(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP

    )

    env.enable_reset_distribution = False
    env.domain_randomization = 0.20  # make DR large to see robustness of PID
    T = 10000

    # == Action Design:
    # a[0]: thrust
    # a[1:3]: roll, pitch, yaw
    actions = np.zeros((T, 4))
    actions[:, 0] = -0.8 # env.drone.HOVER_ACTION - 0.42
    actions[:, 1] = 0.0 #5 * np.sin(np.arange(T)/T*20*np.pi)  # roll
    actions[:, 2] = 0.0 #5 * np.cos(np.arange(T)/T*20*np.pi)  # pitch
    actions[:, 3] = 0  # yaw

    env.reset()
    j = 0
    for i in range(T):
        time.sleep(1/100)
        env.step(actions[i])
        j = j + 1
        if j % 100 == 0:
            j = 0
            # Note: that domain randomization is enabled by default, so the
            # drone will not stay in position but will lift or sink.
            env.reset()
    env.close()


if __name__ == '__main__':
    main()

