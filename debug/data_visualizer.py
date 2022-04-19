r"""Visualization of logged flight data from CrazyFlie drone.

Important Note:
    this file assumes that you are using the CrazyFlie Firmware adopted the the
    Chair of Data Processing - Technical University Munich (TUM)
"""
import pandas as pd
import numpy as np
import torch
import gym
import pybullet as pb
import matplotlib.pyplot as plt
import phoenix_drone_simulation  # noqa
import time
import argparse
import warnings

from phoenix_drone_simulation.utils.trajectory_generator import TrajectoryGenerator


def remove_battery_compensation(
        PWMs: np.ndarray,
        supply_voltage: float
):
    r"""Remove battery compensation gain from PWM voltages."""
    percentage = PWMs / 65536
    volts = percentage * supply_voltage

    a = -0.0006239
    b = 0.088
    c = -volts
    thrust = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    PWMs_cleaned = thrust / 60 * 65536
    return PWMs_cleaned


def create_sim_data(N, env, env_id, actions):
    # _print(f'read from env: {env.unwrapped.init_rpy_dot}')
    render = False
    env.render() if render else None
    x = env.reset()
    print(f'x: {x.shape}')
    # _print(f'read from drone: {env.unwrapped.drone.rpy_dot}')
    print(f'env.obs: {env.observation_space.shape[0]}')
    data_sim = np.zeros((N, env.observation_space.shape[0]))
    data_sim[0] = x
    dones = np.zeros(N)
    thrusts_new = np.zeros((N, 4))
    PWMs = np.zeros((N, 4))
    PWMs_cleaned = np.zeros((N, 4))
    thrusts_new[0] = env.unwrapped.drone.y
    PWMs[0] = env.unwrapped.drone.PWMs

    i = 0
    while i < N-1:
        # action = np.zeros(4)
        action = actions[i]
        obs, r, done, info = env.step(action)
        i += 1
        data_sim[i] = obs  # (x, y, z, a, b, c, d, ... )
        if env_id == 'DroneCircleBulletEnv-v0' or env_id == 'DroneTakeOffBulletEnv-v0':
            thrusts_new[i] = env.unwrapped.drone.y
            PWMs[i] = env.unwrapped.drone.PWMs
        dones[i] = float(done)
        if render:
            time.sleep(0.01)
    # env.close()
    return data_sim, thrusts_new, PWMs


def setup_environment(env_id, df):
    env = gym.make(env_id)
    """ Important Note:
        Write to wrapped env class with: env.unwrapped.attribute = new_value
    """
    if env_id == 'DroneCircleBulletEnv-v0' \
            or env_id == 'DroneTakeOffBulletEnv-v0':
        # Deactivate Domain Randomization and Noise...
        env.unwrapped.observation_noise = -1.
        env.unwrapped.action_noise = -1
        env.unwrapped.use_domain_randomization = False
        env.unwrapped.domain_randomization = -1
        # deactivate
        env.unwrapped.enable_reset_distribution = False
    else:
        raise ValueError

    # Set simulation init state to initially measured real world state
    env.unwrapped.init_xyz = np.array([
        df['x'].values[0], df['y'].values[0], df['z'].values[0]])
    env.unwrapped.init_xyz_dot = np.array([
        df['x_dot'].values[0], df['y_dot'].values[0], df['z_dot'].values[0]])
    env.unwrapped.init_rpy = np.array([
        df['roll'].values[0],  df['pitch'].values[0], df['yaw'].values[0]])
    # print(f'Read init: RPY = {env.unwrapped.init_rpy*180/np.pi}')
    env.unwrapped.init_quaternion = np.array(pb.getQuaternionFromEuler(env.unwrapped.init_rpy))
    # print(f'init: quat: {env.unwrapped.init_quaternion}')
    env.unwrapped.init_rpy_dot = np.array([
        df['roll_dot'].values[0], df['pitch_dot'].values[0], df['yaw_dot'].values[0]])
    env.unwrapped.init_rpy_dot = np.array([
        df['roll_dot'].values[0], df['pitch_dot'].values[0], df['yaw_dot'].values[0]])
    return env


def plot_state_evolution(
        df: pd.DataFrame,
        env_id: str,
        pi,
        obs_rms,
        k,
        plot_sim: bool = True,
        debug=True
):
    """Plot trajectories of real and simulated data."""
    def _print(*msg):
        if debug:
            print(*msg)
    ts = df['time'].values  # time in [s]
    N = ts.shape[0]
    _print('N:', N)
    print(f'Flight time: {ts[-1] - ts[0]:0.2f}s')
    env = setup_environment(env_id, df)

    motor_values_sim = np.zeros((N, 4))
    # motor_values_sim[0] = env.unwrapped.drone.PWMs
    # actions = df[['n0', 'n1', 'n2', 'n3']].values
    pwms = df[['mot0', 'mot1', 'mot2', 'mot3']].to_numpy(dtype=np.float32)


    # Remove battery compenstation
    N = ts.shape[0]
    PWMs_cleaned = np.zeros_like(pwms)
    volts = df[['bat']].to_numpy(dtype=np.float32)
    # print(f'Volts: {volts}')
    for i in range(N):
        PWMs_cleaned[i] = remove_battery_compensation(pwms[i], volts[i])
    actions = PWMs_cleaned / 2 ** 15 - 1

    if plot_sim:
        data_sim, thrusts_new, motor_PWMs = create_sim_data(N, env, env_id, actions)

    M, N = 4, 5
    fig, axes = plt.subplots(M, N, figsize=(8, 6))
    fig.tight_layout()
    ax_num = 0

    """"==== XYZ ===="""
    for i, col in enumerate(['x', 'y', 'z']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        ax.plot(ts,  df[col], label=col.upper()+' (real)')
        ax.plot(ts, data_sim[:, i], label=col.upper()+' (sim)') if plot_sim else None
        ax.grid()
        ax_num += 1

    """"==== RPY ===="""
    ax_num = N
    # abcd_real = df[['a', 'b', 'c', 'd']].values
    rpy_sim = data_sim[:, 3:6] * 180 / np.pi if plot_sim else None

    if plot_sim:
        rpy_sim = np.zeros_like(data_sim[:, 0:3])
        for i, row in enumerate(data_sim[:, 3:7]):  # iterate over quaternions
            # print(f'quat: {row}')
            rpy_sim[i] = np.asarray(pb.getEulerFromQuaternion(row)) * 180 / np.pi

    rpy_real = df[['roll', 'pitch', 'yaw']].values * 180 / np.pi
    for i, col in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        ax.plot(ts, rpy_real[:, i], label=col.upper()+' (real)')
        ax.plot(ts, rpy_sim[:, i], label=col.upper()+' (sim)') if plot_sim else None
        ax.grid()
        ax_num += 1

    """"==== Linear Velocities ===="""
    ax_num = 2 * N
    vel_sim = data_sim[:, 7:10] if plot_sim else None
    for i, col in enumerate(['x_dot', 'y_dot', 'z_dot']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col + ' [m/s]')
        ax.plot(ts, df[col].values, label=col.upper()+' (real)')
        ax.plot(ts, vel_sim[:, i], label=col.upper()+' (sim)')  if plot_sim else None
        ax.grid()
        ax.legend()
        ax_num += 1

    """"==== Attitude Rates ===="""
    ax_num = 3 * N
    for i, col1 in enumerate(['roll_dot', 'pitch_dot', 'yaw_dot']):
    # for i, (col1, col2) in enumerate(zip(['roll_dot', 'pitch_dot', 'yaw_dot'],
    #                                      ['gyro_x', ' gyro_y', ' gyro_z'])):
        ax = axes.flatten()[ax_num]
        ax.set_title(col1 + ' [deg/s]')
        ax.plot(ts, df[col1].values*180/np.pi, label=col1.upper() + ' (real)')
        ax.plot(ts, data_sim[:, i + 10]*180/np.pi, label=col1.upper() + ' (sim)')  if plot_sim else None

    # Calculate Ground Truth RPY dot....
        # gt_rpy_dot = np.zeros_like(df[col].values)
        # dt = 0.01
        # rpy_vals = df[col.replace('d', '')].values  # take roll instead of droll
        # gt_rpy_dot[1:] = (rpy_vals[1:] - rpy_vals[:-1]) * 180 / (np.pi * dt)
        # ax.plot(ts, gt_rpy_dot, label=col.upper() + ' (gt)')
        ax.grid()
        ax.legend()
        ax_num += 1

    """"==== Thrust Plots ===="""
    ax_num = M - 1
    # actions = get_nn_outputs(df, pi, obs_rms=obs_rms)
    # for i in range(4):
    for i, col in enumerate(['n0', 'n1', 'n2', 'n3']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col)
        # ax.set_title(f'Thrusts Motor{i}')
        ax.plot(ts, df[col], label=col.upper() + ' (real)')
        # if plot_sim:
        #     ax.plot(ts, thrusts_new[:, i], label='Molchanov et al')
        # ax.set_ylim(-1.2, 1.2)
        ax.grid()
        ax.legend()
        ax_num += N

    """"==== PWMs ===="""
    ax_num = M
    # actions = get_nn_outputs(df, pi, obs_rms=obs_rms)
    for i, col in enumerate(['mot0', 'mot1', 'mot2', 'mot3']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col)
        ax.plot(ts, df[col], label=col.upper() + ' (real)')
        # motor_label = f'mot{i}'
        # ax.plot(ts, df[motor_label], label=motor_label.upper()+' (real)')
        ax.plot(ts, motor_PWMs[:, i], label=col.upper() + ' (sim)')  if plot_sim else None
        ax.plot(ts, PWMs_cleaned[:, i], label=col.upper() + ' (cleaned)')  if plot_sim else None

        # new: remove battery compensation values...

        ax.set_ylim(0, 2**16)
        ax.legend()
        ax.grid()
        ax_num += N

    plt.savefig(f'/var/tmp/figures_{k}.png')
    plt.show()


def get_nn_outputs(df, pi, obs_rms):
    obs_np = df[['x', 'y', 'z',
                 # 'a', 'b', 'c', 'd',
                 'roll', 'pitch', 'yaw',
                 'x_dot', 'y_dot', 'z_dot',
                 'roll_dot', 'pitch_dot', 'yaw_dot',
                 'n0', 'n1', 'n2', 'n3'
                 ]].values
    # calculate NN outputs
    obs_normed = obs_rms(torch.as_tensor(obs_np, dtype=torch.float32))

    with torch.no_grad():
        ones_vec = torch.ones(16, dtype=torch.float32)
        after_obs = obs_rms(ones_vec)
        final_oputput = pi(after_obs).numpy()
        ones_vec = pi(torch.ones(16, dtype=torch.float32)).numpy()

    with torch.no_grad():
        actions = pi(torch.as_tensor(obs_normed, dtype=torch.float32)).numpy()
    return actions


def plot_neural_network_outputs(
        df: pd.DataFrame,
        pi,
        obs_rms
) -> None:
    """Visualize the outputs of the Neural Networks.

    Parameters
    ----------
    df

    Returns
    -------

    """
    ts = df['time'].values
    actions = get_nn_outputs(df, pi, obs_rms)

    for i, col in enumerate(['n0', 'n1', 'n2', 'n3']):
        plt.subplot(2, 4, i+1)
        plt.title(col)
        plt.plot(ts, df[col], label=col.upper()+' (real)')
        plt.plot(ts, actions[:, i], label=col.upper()+' (sim)')
        plt.ylim(-1.5, 1.5)
        plt.legend()

    def action_to_RPM(action):
        return 45000 + 15000 * np.clip(action, -1, 1)

    # plot differences between real NN outputs and sim NN outputs
    for i, col in enumerate(['mot0', 'mot1', 'mot2', 'mot3']):
        plt.subplot(2, 4, i+5)
        plt.title(col)
        plt.plot(ts, df[col], label=col.upper()+' (real)')
        print(f'col={col}: {df[col].values}')
        plt.plot(ts, action_to_RPM(actions[:, i]), label=col.upper()+' (sim)')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--logdir', type=str, required=True,
                        help='Define a custom directory for logging.')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print debug infos to console.')
    parser.add_argument('--env', type=str, default='DroneCircleBulletEnv-v0',
                        help='Define to environment to be evaluated.')
    file_name_path = '/Users/sven/Documents/Project-Phoenix/Real-world-Tests/2021_KW_20/Logs/KW_20_model_PID_2.json'
    parser.add_argument('--model', type=str, default=file_name_path,
                        help='Define a custom directory for logging.')
    parser.add_argument('--no-sim', '-ns', action='store_true',
                        help='Print debug infos to console.')
    parser.add_argument('--sim', '-s', type=str, default='pb',
                        help='Choose simulator: pb, quad')
    args, unparsed_args = parser.parse_known_args()

    df = pd.read_csv(args.logdir)

    generator = TrajectoryGenerator(args.env, debug=args.debug)
    generator.load_file_from_disk(args.model)

    # plot_neural_network_outputs(df, generator.policy_net, generator.obs_rms)

    plot_state_evolution(
        df,
        args.env,
        generator.policy_net,
        generator.obs_rms,
        k=0,
        plot_sim=not args.no_sim,
        debug=args.debug)
