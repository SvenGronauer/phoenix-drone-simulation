""" PyTorch implementation of DDPG Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    04.05.2022

Copied and adapted from Spinning Up:
https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/ddpg/core.py
"""
from copy import deepcopy
import numpy as np
from torch.optim import Adam
import gym
import time
import torch
import torch.nn as nn


# local imports
from phoenix_drone_simulation.algs.ddpg.buffer import ReplayBuffer
import phoenix_drone_simulation.algs.core as core
from phoenix_drone_simulation.utils import loggers
import phoenix_drone_simulation.utils.mpi_tools as mpi
from phoenix_drone_simulation.utils import utils
import phoenix_drone_simulation.algs.utils as alg_utils


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        # self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.pi = core.build_mlp_network(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.q = core.build_mlp_network([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 ac_kwargs,
                 # hidden_sizes=(256,256),
                 # activation=nn.ReLU
                 ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(
            obs_dim, act_dim,
            act_limit=act_limit,
            **ac_kwargs["pi"]
            # hidden_sizes=ac_kwargs["pi"]["hidden_sizes"],
            # activation,
        )
        self.q = MLPQFunction(
            obs_dim, act_dim,
            # hidden_sizes, activation
            **ac_kwargs["q"]
        )

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

    def forward(self, obs: torch.Tensor):
        return self.act(obs), None


class DeepDeterministicPolciyGradientAlgorithm(core.OffPolicyGradientAlgorithm):
    def __init__(
            self,
            actor: str,
            env_id: str,
            logger_kwargs: dict,
            ac_kwargs=dict(),
            alg='ddpg',
            batch_size=1000*mpi.num_procs(),
            buffer_size=int(1e6),
            check_freq: int = 25,
            epochs=100,
            gamma=0.99,
            polyak=0.995,
            pi_lr=1e-4,
            q_lr=1e-3,
            mini_batch_size=128,
            warmup_steps=10000,
            update_after=1000,
            update_every=50,
            act_noise=0.1,
            num_test_episodes=10,
            max_ep_len=1000,
            save_freq=10,
            seed=0,
            **kwargs  # use to log parameters from child classes
    ):
        """
        Deep Deterministic Policy Gradient (DDPG)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to DDPG.

            seed (int): Seed for random number generators.

            batch_size (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            buffer_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.

            mini_batch_size (int): Minibatch size for SGD.

            warmup_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            act_noise (float): Stddev for Gaussian exploration noise added to
                policy at training time. (At test time, no noise is added.)

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        self.params = locals()

        self.actor = actor  # todo: meaningless at the momment
        self.act_noise = act_noise
        self.check_freq = check_freq
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.epochs = epochs
        self.gamma = gamma
        self.local_batch_size = batch_size // mpi.num_procs()
        self.polyak = polyak
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.in_warm_up = True
        self.warmup_steps = warmup_steps // mpi.num_procs()

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

        self.act_limit = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # === set up MPI specifics
        self._init_mpi()

        # Experience buffer
        self.buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size//mpi.num_procs()
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        loggers.info('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        # Set up model saving
        self.logger.setup_torch_saver(self.ac)
        self.logger.torch_save()

        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
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

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi, 'Q': self.ac.q}
            for key, module in modules.items():
                flat_params = alg_utils.get_flat_params_from(module).numpy()
                global_min = mpi.mpi_min(np.sum(flat_params))
                global_max = mpi.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_q(self, data):
        r"""Set up function for computing DDPG Q-loss"""
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], \
                         data['done']

        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    def compute_loss_pi(self, data):
        r"""Set up function for computing DDPG pi loss"""
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def get_action(self, o: np.ndarray, noise_scale: float):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

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

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('QVals', min_and_max=True)
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

        for t in range(self.local_batch_size):
            self.in_warm_up = True if len(self.buffer) < self.warmup_steps else False
            if self.in_warm_up:
                # Until warmup_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards,
                # use the learned policy (with some noise, via act_noise).
                a = self.env.action_space.sample()
            else:
                a = self.get_action(o, self.act_noise)

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
            self.logger.store(QVals=0., LossQ=0.0, LossPi=0.0)

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        mpi.mpi_avg_grads(self.ac.q)  # average grads across MPI processes
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


def get_alg(env_id, **kwargs) -> DeepDeterministicPolciyGradientAlgorithm:
    return DeepDeterministicPolciyGradientAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='ddpg', env_id=env_id)
    defaults.update(**kwargs)
    alg = DeepDeterministicPolciyGradientAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()

    return ac, env
