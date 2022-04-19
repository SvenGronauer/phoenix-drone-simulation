r"""Evaluation of RL agent performance.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    14.04.2022
"""
import numpy as np
import os
import gym
import torch
import atexit
import phoenix_drone_simulation.utils.mpi_tools as mpi_tools



class EnvironmentEvaluator(object):
    def __init__(self, log_dir, log_costs=True):

        self.log_dir = log_dir
        self.env = None
        self.ac = None
        self.log_costs = log_costs

        # open returns.csv file at the beginning to avoid disk access errors
        if mpi_tools.proc_id() == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.ret_file_name = 'returns.csv'
            self.ret_file = open(os.path.join(log_dir, self.ret_file_name), 'w')
            # Register close function is executed for normal program termination
            atexit.register(self.ret_file.close)
            if log_costs:
                self.c_file_name = 'costs.csv'
                self.costs_file = open(os.path.join(log_dir, self.c_file_name), 'w')
                atexit.register(self.costs_file.close)
        else:
            self.ret_file_name = None
            self.ret_file = None
            if log_costs:
                self.c_file_name = None
                self.costs_file = None

    def close(self):
        """Close opened output files.

        Note this is done immediately after training in order to avoid possible
        OS errors.
        """
        if mpi_tools.proc_id() == 0:
            self.ret_file.close()
            if self.log_costs:
                self.costs_file.close()

    def eval(self, env, ac, num_evaluations):
        r""" Evaluate actor critic module for given number of evaluations."""
        self.ac = ac
        self.ac.eval()  # disable exploration noise

        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
        else:
            raise TypeError('Env is not of type: str, gym.Env')

        size = mpi_tools.num_procs()
        num_local_evaluations = num_evaluations // size
        returns = np.zeros(num_local_evaluations, dtype=np.float32)
        costs = np.zeros(num_local_evaluations, dtype=np.float32)
        ep_lengths = np.zeros(num_local_evaluations, dtype=np.float32)

        for i in range(num_local_evaluations):
            returns[i], ep_lengths[i], costs[i] = self.eval_once()
        # Gather returns from all processes
        # Note: only root process owns valid data...
        returns = list(mpi_tools.gather_and_stack(returns))
        costs = list(mpi_tools.gather_and_stack(costs))

        # now write returns as column into output file...
        if mpi_tools.proc_id() == 0:
            self.write_to_file(self.ret_file, contents=returns)
            print('Saved to:', os.path.join(self.log_dir, self.ret_file_name))
            if self.log_costs:
                self.write_to_file(self.costs_file, contents=costs)
            print(f'Mean Ret: { np.mean(returns)} \t'
                  f'Mean EpLen: {np.mean(ep_lengths)} \t'
                  f'Mean Costs: {np.mean(costs)}')

        self.ac.train()  # back to train mode
        return np.array(returns), np.array(ep_lengths), np.array(costs)

    def eval_once(self):
        assert not self.ac.training, 'Call actor_critic.eval() beforehand.'
        done = False
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0

        while not done:
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, *_ = self.ac(obs)
            x, r, done, info = self.env.step(action)
            ret += r
            costs += info.get('cost', 0.)
            episode_length += 1

        return ret, episode_length, costs

    @staticmethod
    def write_to_file(file, contents: list):
        if mpi_tools.proc_id() == 0:
            column = [str(x) for x in contents]
            file.write("\n".join(column) + "\n")
            file.flush()
