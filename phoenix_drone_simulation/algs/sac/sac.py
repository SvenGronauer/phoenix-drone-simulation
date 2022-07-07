""" PyTorch implementation of Soft Actor Critic (SAC) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    07.07.2022

based on the Spinning Up implementation:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
"""
from copy import deepcopy
import itertools
import gym
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.normal import Normal

# local imports
import phoenix_drone_simulation.utils.mpi_tools as mpi
from phoenix_drone_simulation.utils import loggers
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.utils import utils


from phoenix_drone_simulation.algs.sac.buffer import ReplayBuffer
import phoenix_drone_simulation.algs.utils as U

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        network_sizes = [obs_dim] + list(hidden_sizes)
        self.net = core.build_mlp_network(
            network_sizes, activation, output_activation=activation
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        q_network_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q = core.build_mlp_network(q_network_sizes, activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 ac_kwargs: dict,
                 # hidden_sizes=(256, 256),
                 # activation=nn.ReLU
                 ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, **ac_kwargs["pi"]
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, **ac_kwargs["q"])
        self.q2 = MLPQFunction(obs_dim, act_dim, **ac_kwargs["q"])

    def act(self, obs: torch.Tensor, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

    def forward(self, obs) -> tuple:
        """Make the interface of this method equal to ActorCritics of On-Policy algorithms."""
        a = self.act(obs, deterministic=True)

        # same return tuple as: return a.numpy(), v.numpy(), logp_a.numpy()
        return a, None, None


class SoftActorCriticAlgorithm(core.OffPolicyGradientAlgorithm):
    def __init__(
            self,
            env_id,
            logger_kwargs: dict,
            actor='mlp',  # meaningless at the moment
            ac_kwargs=dict(),
            alg='sac',
            alpha=0.2,
            batch_size=1000 * mpi.num_procs(),
            buffer_size=int(1e6),
            check_freq: int = 25,
            epochs=100,
            gamma=0.99,
            lr=1e-3,
            max_ep_len=1000,
            mini_batch_size=64,
            polyak=0.995,
            save_freq=1,
            seed=0,
            start_steps=10000,
            update_after=1000,
            update_every=50,
            **kwargs  # use to log parameters from child classes
    ):
        assert 0 < polyak < 1
        self.params = locals()

        self.alg = alg
        self.alpha = alpha
        self.batch_size = batch_size
        self.check_freq = check_freq
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.epochs = epochs
        self.gamma = gamma
        self.local_batch_size = batch_size // mpi.num_procs()
        self.polyak = polyak
        self.save_freq = save_freq
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.in_warm_up = True

        # Note: NEW: call gym.make with **kwargs (to allow customization)
        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors
        if hasattr(self.env, '_max_episode_steps'):
            # Collect information from environment if it has an time wrapper
            self.max_ep_len = self.env._max_episode_steps
        else:
            self.max_ep_len = 1000
        self.act_limit = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]
        # ==== Call assertions....
        self._sanity_checks()

        # === Set up logger and save configuration to disk
        self.logger_kwargs = logger_kwargs
        self.logger = self._init_logger()
        self.logger.save_config(self.params)
        # save environment settings to disk
        self.logger.save_env_config(env=self.env)
        loggers.set_level(loggers.INFO)

        # === Seeding
        seed += 10000 * mpi.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed=seed)

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        self.freeze(self.ac_targ.parameters())

        # === set up MPI specifics and sync parameters across all processes
        self._init_mpi()

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters()
        )
        # Experience buffer
        self.buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size//mpi.num_procs()
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        loggers.info( '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_torch_saver(self.ac)
        self.logger.torch_save()


        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.logger.info('Done with initialization.')

    def _init_logger(self):
        # pop to avoid self object errors
        self.params.pop('self')
        # move nested kwargs to highest dict level
        if 'kwargs' in self.params:
            self.params.update(**self.params.pop('kwargs'))
        logger = loggers.EpochLogger(**self.logger_kwargs)
        return logger

    def _init_mpi(self) -> None:
        """ Initialize MPI specifics

        Returns
        -------

        """
        if mpi.num_procs() > 1:
            loggers.info(f'Started MPI with {mpi.num_procs()} processes.')
            # Avoid slowdowns from PyTorch + MPI combo.
            mpi.setup_torch_for_mpi()
            dt = time.time()
            loggers.info('Sync actor critic parameters...')
            # Sync params across cores: only once necessary, grads are averaged!
            mpi.sync_params(self.ac)
            self.ac_targ = deepcopy(self.ac)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')

    def _sanity_checks(self):
        assert self.max_ep_len <= self.local_batch_size, \
            f'Reduce number of cores ({mpi.num_procs()}) or increase ' \
            f'batch size {self.batch_size}.'
        assert self.local_mini_batch_size > 0, f"Please increase batch size"
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi, 'Q1': self.ac.q1, 'Q2': self.ac.q2}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi.mpi_min(np.sum(flat_params))
                global_max = mpi.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], \
                         data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    @staticmethod
    def freeze(params):
        for p in params:
            p.requires_grad = False

    @staticmethod
    def unfreeze(params):
        for p in params:
            p.requires_grad = True

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for self.epoch in range(self.epochs):
            self.epoch_time = time.time()
            is_last_epoch = self.epoch == self.epochs - 1
            # Note: update function is called during rollouts!
            self.roll_out()
            # Save (or print) information about epoch
            self.log(self.epoch)

            # Check if all models own the same parameter values
            if self.epoch % self.check_freq == 0:
                self.check_distributed_parameters()

            # Save model to disk
            if is_last_epoch or self.epoch % self.save_freq == 0:
                self.logger.save_state(state_dict={}, itr=None)

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac, self.env

    def log(self, epoch):
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.batch_size
        fps = self.batch_size / (time.time() - self.epoch_time)

        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('Q1Vals', min_and_max=True)
        self.logger.log_tabular('Q2Vals', min_and_max=True)
        self.logger.log_tabular('LogPi', std=False)
        self.logger.log_tabular('LossPi', std=False)
        self.logger.log_tabular('LossQ', std=False)
        self.logger.log_tabular('InWarmUp', float(self.in_warm_up))
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))
        self.logger.dump_tabular()

    def roll_out(self):
        r"""Rollout >>one<< episode and store to buffer."""

        o, ep_ret, ep_len = self.env.reset(), 0., 0
        local_start_steps = int(self.start_steps / mpi.num_procs())

        for t in range(self.local_batch_size):
            self.in_warm_up = True if len(self.buffer) < local_start_steps else False
            if self.in_warm_up:
                # Until warmup_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards,
                # use the learned policy (with some noise, via act_noise).
                a = self.env.action_space.sample()
            else:
                a = self.get_action(o, deterministic=False)

            next_o, r, done, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            terminal = False if ep_len == self.max_ep_len else done

            # Store experience to replay buffer
            self.buffer.store(o, a, r, next_o, terminal)

            # Update handling
            if not self.in_warm_up and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.buffer.sample_batch(self.local_mini_batch_size)
                    self.update(data=batch)

            o = next_o
            timeout = (ep_len == self.max_ep_len)
            if timeout or done:
                 # only save EpRet / EpLen if trajectory finished
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0., 0
        if self.in_warm_up:
            # add zero values to prevent logging errors during warm-up
            self.logger.store(Q1Vals=0, Q2Vals=0, LossQ=0, LossPi=0, LogPi=0)

    def update(self, data: dict):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        mpi.mpi_avg_grads(self.ac.q1)  # average grads across MPI processes
        mpi.mpi_avg_grads(self.ac.q2)  # average grads across MPI processes
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.freeze(self.q_params)

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.unfreeze(self.q_params)

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


def get_alg(env_id, **kwargs) -> core.Algorithm:
    return SoftActorCriticAlgorithm(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='sac', env_id=env_id)
    defaults.update(**kwargs)
    alg = SoftActorCriticAlgorithm(
        env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env
