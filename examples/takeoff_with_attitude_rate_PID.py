"""Use an Attitude-Rate Controller to stabilize the drone for Take-off task.

Note: the thrust of the drone is hard-coded.

Author:     Sven Gronauer
Created:    19.04.2022
"""
import time
import numpy as np

# local imports
from phoenix_drone_simulation.envs.takeoff import DroneTakeOffBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate


def main():
    # control mode of DroneTakeOffBulletEnv is PWM by default:
    env = DroneTakeOffBulletEnv()
    # Important: call render before updating the controller and domain rand
    env.render()

    # overwrite PWM control with Attitude-Rate PID controller
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP

    )

    env.enable_reset_distribution = True
    # set domain randomization (DR) to zero such that all motors have same
    # properties
    env.domain_randomization = 0.0
    T = 10000

    # == Action Design:
    # a[0]: thrust
    # a[1:3]: roll_dot, pitch_dot, yaw_dot
    actions = np.zeros((T, 4))
    actions[:, 0] = env.drone.HOVER_ACTION + 0.2  # thrust
    actions[:, 1] = 0  # roll_dot
    actions[:, 2] = 0  # pitch_dot
    actions[:, 3] = 0  # yaw_dot

    env.reset()
    j = 0
    while True:
        time.sleep(1/100)
        obs, reward, done, info = env.step(actions[j])
        j += 1
        if done or j%150==0:
            j = 0
            env.reset()
    # env.close()


if __name__ == '__main__':
    main()

