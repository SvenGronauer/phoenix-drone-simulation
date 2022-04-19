r"""Generate batch-wise trajectory data from a drone environment.

Author:     Sven Gronauer
"""
import torch
import gym
import numpy as np
import time

# local imports
import phoenix_drone_simulation  # noqa
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.utils.utils import load_network_json, get_file_contents, get_policy_filename_path
from phoenix_drone_simulation.utils.online_mean_std import OnlineMeanStd


def get_generator(
        env_name: str,
        file_name: str = 'model_50_50_relu_PWM_circle_task.json'
):
    generator = TrajectoryGenerator(env_name)
    fnp = get_policy_filename_path(file_name)
    generator.load_file_from_disk(fnp)
    return generator


class TrajectoryGenerator:
    def __init__(
            self,
            env_id: str,
            debug: bool = True,
    ):

        self.env = gym.make(env_id)
        self.policy_net = None
        self.obs_rms = None
        self.debug = debug

    def evaluate(self, num_trajectories=32, return_stds=False, return_raw=False):
        msg = 'Load model from disk before evaluation. Call load_file_from_disk'
        assert self.policy_net is not None, msg
        returns = []

        self.policy_net.eval()
        for i in range(num_trajectories):
            # run one trajectory
            ret, trajectory_length = self.evaluate_once()
            returns.append(ret)
        if return_raw:
            return np.array(returns)
        else:
            if return_stds:
                return float(np.mean(returns)), float(np.std(returns))
            else:
                return float(np.mean(returns))

    def evaluate_once(self, render=False) -> tuple:
        """ Evaluates the policy for one trajectory.
        If render is true, PyBullet GUI is used for visualization.

        """
        TARGET_FPS = 100
        target_dt = 1.0 / TARGET_FPS
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        done = False
        while not done:
            ts = time.time()
            self.env.render(mode='human') if render else None
            x = torch.as_tensor(x, dtype=torch.float32)
            x_stand = self.obs_rms(x)
            with torch.no_grad():
                action = self.policy_net(x_stand).numpy()
            x, r, done, info = self.env.step(action)
            # print(f'Action={action}')
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            delta = time.time() - ts
            if render and delta < target_dt:
                time.sleep(target_dt-delta)  # sleep delta time
        if render:
            print(f'Return: {ret}\t Length: {episode_length}\t Costs:{costs}')
        return ret, episode_length

    def get_batch(self, N=5000) -> tuple:
        """Generates a batch of data based on the current policy.

        Note: The drone environments produce trajectories of length 500 and then
        reset their internal status.

        Returns
        -------
        tuple, holding (X, Y)
            X is data matrix of size NxD, where D is dimension of observations
            Y is data matrix of size NxD, where D is dimension of observations

        """
        msg = 'Load model from disk before evaluation. Call load_file_from_disk'
        assert self.policy_net is not None, msg
        t = 0
        X = []
        Y = []
        obs = self.env.reset()
        while t < N:
            x = self.obs_rms(torch.as_tensor(obs, dtype=torch.float32))
            X.append(x.numpy())
            with torch.no_grad():
                action = self.policy_net(x).numpy()
            y, r, done, info = self.env.step(action)
            Y.append(y)
            obs = y  # set old state (x_t) as new state (x_{t+1})
            t += 1
            if done:
                obs = self.env.reset()
        return np.array(X), np.array(Y)

    def load_file_from_disk(
            self,
            file_name_path: str
    ):
        # === Load policy as PyTorch module
        policy_net = load_network_json(file_name_path=file_name_path)
        print(policy_net)
        self.policy_net = policy_net

        # === Load observation standardization module..
        data = get_file_contents(file_name_path)
        scaling_parameters = np.array(data['scaling_parameters'])
        obs_oms = OnlineMeanStd(shape=self.env.observation_space.shape)

        obs_oms.mean.data = torch.Tensor(scaling_parameters[0])
        obs_oms.std.data = torch.Tensor(scaling_parameters[1])
        if loggers.MIN_LEVEL <= loggers.DEBUG:
            print('='*55)
            print(f'obs.mean: {obs_oms.mean}')
            print(f'obs.std: {obs_oms.std}')

        self.obs_rms = obs_oms

    def play_policy(self, noise=False):
        """ Render the policy performance in environment...

        noise:
            If true, enable stochastic action selection.
        """
        if not noise:
            self.policy_net.eval()  # Set in evaluation mode before playing
        # i = 0
        # pb.setRealTimeSimulation(1)
        while True:
            self.env.render()
            self.evaluate_once(render=True)
