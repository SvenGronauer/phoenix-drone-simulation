import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

# local imports
import phoenix_drone_simulation  # noqa
from phoenix_drone_simulation.utils import utils


def plot_drone_flight(df, env, rpm=None, debug=True):
    """Display simulated data."""
    def _print(*msg):
        if debug:
            print(*msg)
    N = len(df)
    _print('N:', N)
    ts = np.arange(N) * env.time_step  # [s]

    M, N = 4, 5
    fig, axes = plt.subplots(M, N, figsize=(8, 6))
    fig.tight_layout()
    ax_num = 0

    """"==== XYZ ===="""
    for i, col in enumerate(['x', 'y', 'z']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        print(df[col].values.shape)
        ax.plot(ts,  df[col], label=col.upper()+' (sim)')
        # ax.plot(ts, data_sim[:, i], label=col.upper()+' (sim)')
        ax.grid()
        ax_num += 1

    """"==== RPY ===="""
    ax_num = N
    rpy_real = df[['roll', 'pitch', 'yaw']].values * 180 / np.pi
    for i, col in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col.upper())
        ax.plot(ts, rpy_real[:, i], label=col.upper()+' (sim)')
        ax.grid()
        ax_num += 1
    """"==== Linear Velocities ===="""
    ax_num = 2 * N
    for i, col in enumerate(['x_dot', 'y_dot', 'z_dot']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col + ' [m/s]')
        ax.plot(ts, df[col].values, label=col.upper()+' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1
    """"==== Attitude Rates ===="""
    ax_num = 3 * N
    for i, col in enumerate(['roll_dot', 'pitch_dot', 'yaw_dot']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col + ' [deg/s]')
        ax.plot(ts, df[col].values*180/np.pi, label=col.upper() + ' (sim)')
        ax.grid()
        ax.legend()
        ax_num += 1
    """"==== Actions===="""
    ax_num = M - 1
    for i, col in enumerate(['n0', 'n1', 'n2', 'n3']):
        ax = axes.flatten()[ax_num]
        ax.set_title(col)
        ax.plot(ts, df[col], label=col.upper() + ' (sim)')
        ax.set_ylim(-1.2, 1.2)
        ax.grid()
        ax.legend()
        ax_num += N
    # """"==== PWMs ===="""
    ax_num = M
    # actions = get_nn_outputs(df, pi, obs_rms=obs_rms)
    for i, col in enumerate(['n0', 'n1', 'n2', 'n3']):
        ax = axes.flatten()[ax_num]
        ax.set_title(f'PWM {i}')
        ax.plot(ts, np.clip(0.5*(df[col]+1), 0, 1), label=col.upper() + ' (sim)')
        if rpm is not None:
            ax.plot(ts, rpm[:len(ts), i], label='RPM (sim)')
        # motor_label = f'mot{i}'
        # ax.plot(ts, df[motor_label], label=motor_label.upper()+' (real)')
        # ax.plot(ts, motor_values_sim[:, i], label=col.upper() + ' (sim)')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid()
        ax_num += N

    # plt.savefig(f'/var/tmp/figures_{k}.png')
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


def convert_simulation_obs_to_data_frame(obs: np.ndarray) -> pd.DataFrame:
    cols = [
        'x', 'y', 'z',
        'roll', 'pitch', 'yaw',
        'x_dot', 'y_dot', 'z_dot',
        'roll_dot', 'pitch_dot', 'yaw_dot',
        'n0', 'n1', 'n2', 'n3']
    df = pd.DataFrame(data=obs, columns=cols)
    return df


def play_after_training(actor_critic, env, noise=False):
    def convert_quat_obs_to_rpy(_x):
        x = np.zeros(16)
        x[0:3] = _x[:3]
        x[3:6] = np.asarray(env.bc.getEulerFromQuaternion(_x[3:7]))
        x[6:16] = _x[7:17]
        return x

    if not noise:
        actor_critic.eval()  # Set in evaluation mode before playing
    actions = []
    done = False
    x = env.reset()
    xs = [convert_quat_obs_to_rpy(x), ]
    drone_rpms = [env.unwrapped.drone.x, ]
    # while not done:
    T = 500
    for t in range(T):
        obs = torch.as_tensor(x, dtype=torch.float32)
        action, *_ = actor_critic(obs)
        actions.append(action)
        x, r, done, info = env.step(action)
        drone_rpms.append(env.unwrapped.drone.x)
        xs.append(convert_quat_obs_to_rpy(x))
    return np.array(xs), np.array(drone_rpms)


def main(args):
    # load actor-critic and environment from disk
    ac, env = utils.load_actor_critic_and_env_from_disk(args.ckpt)

    # adjustments to env...
    # Deactivate Domain Randomization and Noise...
    # env.unwrapped.observation_noise = -1.
    env.unwrapped.action_noise = 0.03
    env.unwrapped.use_domain_randomization = False
    env.unwrapped.domain_randomization = -1

    obs, RPMs = play_after_training(actor_critic=ac, env=env, noise=False)

    df = convert_simulation_obs_to_data_frame(obs=obs)
    print(df)

    plot_drone_flight(df, env, RPMs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, default=None, required=True,
                        help='Choose from: {ppo, trpo}')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print debug infos to console.')
    parser.add_argument('--env', type=str, default='DroneCircleBulletEnv-v0',
                        help='Example: DroneCircleBulletEnv-v0')
    args, unparsed_args = parser.parse_known_args()
    main(args)

