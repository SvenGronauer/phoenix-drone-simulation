""" Simple PyTorch implementation of Importance-weighted Policy Gradient (IWPG)

Author:     Sven Gronauer (sven.gronauer@tum.de)
based on:   Spinning Up's Repository
            https://spinningup.openai.com/en/latest/

            Successful Ingredients of Policy Gradient Algorithms
            https://github.com/SvenGronauer/successful-ingredients-paper
"""
import numpy as np
import gym
import time
import torch
import random
from copy import deepcopy

# local imports
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.utils import utils
import phoenix_drone_simulation.utils.mpi_tools as mpi_tools
import phoenix_drone_simulation.algs.utils as U
import phoenix_drone_simulation.utils.loggers as loggers


class IWPGAlgorithm(core.OnPolicyGradientAlgorithm):
    def __init__(
            self,
            actor: str,
            ac_kwargs: dict,
            critic: str,
            env_id: str,
            epochs: int,
            logger_kwargs: dict,
            adv_estimation_method: str = 'gae',
            alg='iwpg',
            check_freq: int = 25,
            entropy_coef: float = 0.01,
            gamma: float = 0.99,
            lam: float = 0.95,  # GAE scalar
            max_ep_len: int = 1000,
            max_grad_norm: float = 0.5,
            num_mini_batches: int = 16,  # used for value network training
            optimizer: str = 'Adam',  # policy optimizer
            pi_lr: float = 3e-4,
            seq_len: int = 32,  # Splits path into smaller sequences
            seq_overlap: int = 16,  # Sequences overlap e.g. 16 timesteps
            steps_per_epoch: int = 32 * 1000,  # number global steps per epoch
            target_kl: float = 0.01,
            train_pi_iterations: int = 80,
            train_v_iterations: int = 5,
            trust_region='plain',  # used for easy filtering in plot utils
            use_entropy: bool = False,
            use_exploration_noise_anneal: bool = True,
            use_kl_early_stopping: bool = False,
            use_linear_lr_decay: bool = True,
            use_max_grad_norm: bool = False,
            use_reward_scaling: bool = False,
            use_standardized_advantages: bool = False,
            use_standardized_obs: bool = True,
            verbose: bool = True,
            vf_lr: float = 1e-3,
            weight_initialization: str = 'kaiming_uniform',
            save_freq: int = 10,
            seed: int = 0,
            video_freq: int = -1,  # set to positive integer for video recording
            **kwargs  # use to log parameters from child classes
    ):

        # get local parameters before logger instance to avoid unnecessary print
        self.params = locals()

        # Environment calls
        # Note: NEW: call gym.make with **kwargs (to allow customization)
        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors

        # Collect information from environment if it has an time wrapper
        if hasattr(self.env, '_max_episode_steps'):
            max_ep_len = self.env._max_episode_steps

        self.adv_estimation_method = adv_estimation_method
        self.alg = alg
        self.check_freq = check_freq
        self.entropy_coef = entropy_coef if use_entropy else 0.0
        self.epoch = 0  # iterated in learn method
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.local_steps_per_epoch = steps_per_epoch // mpi_tools.num_procs()
        self.logger_kwargs = logger_kwargs
        self.max_ep_len = max_ep_len
        self.max_grad_norm = max_grad_norm
        self.num_mini_batches = num_mini_batches
        self.pi_lr = pi_lr
        self.save_freq = save_freq
        self.seed = seed
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap
        self.steps_per_epoch = steps_per_epoch
        self.target_kl = target_kl
        self.train_pi_iterations = train_pi_iterations
        self.train_v_iterations = train_v_iterations
        self.use_exploration_noise_anneal = use_exploration_noise_anneal
        self.use_kl_early_stopping = use_kl_early_stopping
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_max_grad_norm = use_max_grad_norm
        self.use_reward_scaling = use_reward_scaling
        self.use_standardized_obs = use_standardized_obs
        self.use_standardized_advantages = use_standardized_advantages
        self.video_freq = video_freq
        self.vf_lr = vf_lr

        # ==== Call assertions....
        self._sanity_checks()

        # === Set up logger and save configuration to disk

        self.logger = self._init_logger()
        self.logger.save_config(self.params)
        # save environment settings to disk
        self.logger.save_env_config(env=self.env)
        loggers.set_level(loggers.INFO)

        # === Seeding
        seed += 10000 * mpi_tools.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed=seed)

        # === Setup actor-critic module
        self.ac = core.ActorCritic(
            actor_type=actor,
            critic_type=critic,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            use_standardized_obs=use_standardized_obs,
            use_scaled_rewards=use_reward_scaling,
            weight_initialization=weight_initialization,
            ac_kwargs=ac_kwargs
        )

        # === set up MPI specifics
        self._init_mpi()

        # === Set up experience buffer
        self.buf = core.Buffer(
            actor_critic=self.ac,
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=self.local_steps_per_epoch,
            gamma=gamma,
            lam=lam,
            adv_estimation_method=adv_estimation_method,
            use_scaled_rewards=use_reward_scaling,
            standardize_env_obs=use_standardized_obs,
            standardize_advantages=use_standardized_advantages,
        )

        # Set up optimizers for policy and value function
        self.pi_optimizer = core.get_optimizer(optimizer, module=self.ac.pi,
                                               lr=pi_lr)
        self.vf_optimizer = core.get_optimizer('Adam', module=self.ac.v,
                                               lr=vf_lr)
        # setup scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()

        # Set up model saving
        self.logger.setup_torch_saver(self.ac)
        self.logger.torch_save()

        # setup statistics
        self.best_ep_ret = -float('inf')
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.logger.info('Start with training.')

    def _init_learning_rate_scheduler(self):
        scheduler = None
        if self.use_linear_lr_decay:
            import torch.optim
            def lm(epoch): return 1 - epoch / self.epochs  # linear anneal
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer,
                lr_lambda=lm
            )
        return scheduler

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
        if mpi_tools.num_procs() > 1:
            loggers.info(f'Started MPI with {mpi_tools.num_procs()} processes.')
            # Avoid slowdowns from PyTorch + MPI combo.
            mpi_tools.setup_torch_for_mpi()
            dt = time.time()
            loggers.info('Sync actor critic parameters...')
            # Sync params across cores: only once necessary, grads are averaged!
            mpi_tools.sync_params(self.ac)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')

    def _sanity_checks(self):
        """ Do assertions..."""
        assert self.steps_per_epoch % mpi_tools.num_procs() == 0
        assert self.max_ep_len <= self.local_steps_per_epoch, \
            f'Reduce number of cores ({mpi_tools.num_procs()}) or increase ' \
            f'batch size {self.steps_per_epoch}.'
        assert self.train_pi_iterations > 0
        assert self.train_v_iterations > 0
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi_tools.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi.net, 'Value': self.ac.v.net}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi_tools.mpi_min(np.sum(flat_params))
                global_max = mpi_tools.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_pi(self, data) -> tuple:
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        mask_mean = data['mask'].mean()
        adv_masked = (data['adv'] * data['mask'])

        loss_pi = -(ratio * adv_masked).mean() / mask_mean
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret, mask) -> torch.Tensor:
        r"""Set up function for computing value loss."""
        obs_m = obs * torch.unsqueeze(mask, -1)
        ret_m = ret * mask
        mask_mean = mask.mean()
        # Divide by masks mean (only 0 and 1 entries), too
        return ((self.ac.v(obs_m) - ret_m) ** 2).mean() / mask_mean

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for self.epoch in range(self.epochs):
            self.learn_one_epoch()

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac, self.env

    def learn_one_epoch(self):
        self.epoch_time = time.time()
        is_last_epoch = self.epoch == self.epochs - 1

        if self.use_exploration_noise_anneal:  # update internals of AC
            self.ac.update(frac=self.epoch / self.epochs)
        self.roll_out()  # collect data and store to buffer
        # Save old policy if it achieved best performance so far
        self.log_before_update(self.epoch)
        # Perform policy + value function updates
        # Also updates running statistics
        self.update()
        # Save (or print) information about epoch
        self.log(self.epoch)
        # Check if all models own the same parameter values
        if self.epoch % self.check_freq == 0:
            self.check_distributed_parameters()
        # Save model to disk
        if is_last_epoch or self.epoch % self.save_freq == 0:
            self.logger.save_state(state_dict={}, itr=None)

    def log(self, epoch: int) -> None:
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.steps_per_epoch
        fps = self.steps_per_epoch / (time.time() - self.epoch_time)
        if self.scheduler and self.use_linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()  # step the scheduler if provided
        else:
            current_lr = self.pi_lr

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('Values/Adv', min_and_max=True)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('Misc/Seed', self.seed)
        self.logger.log_tabular('PolicyRatio')
        self.logger.log_tabular('LR', current_lr)
        if self.use_reward_scaling:
            reward_scale_mean = self.ac.ret_oms.mean.item()
            reward_scale_stddev = self.ac.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.use_exploration_noise_anneal:
            noise_std = np.exp(self.ac.pi.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        # some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()

    def log_before_update(self, epoch: int) -> None:
        """ Call this method after roll_out() and before update() """
        ep_ret_stats = self.logger.get_stats('EpRet')
        # Find overall best model
        ep_ret_mean = ep_ret_stats[0]
        ep_ret_max = max(ep_ret_mean, self.best_ep_ret)
        if ep_ret_mean >= self.best_ep_ret:
            self.best_ep_ret = ep_ret_mean
            # Save overall best model (was trained in pervious epoch!)
            self.logger.info(f"Save new best model from epoch {epoch} ({ep_ret_max})")
            self.logger.save_state(state_dict={}, itr=None)

    def pre_process_data(self, raw_data: dict) -> dict:
        """ Pre-process data, e.g. standardize observations, re-scale rewards if
            enabled by arguments.

        Parameters
        ----------
        raw_data
            dictionary holding information obtain from environment interactions

        Returns
        -------
        dict
            holding pre-processed data, i.e. observations and rewards
        """
        data = deepcopy(raw_data)

        if self.use_standardized_obs:
            assert 'obs' in data
            obs = data['obs']
            data['obs'] = self.ac.obs_oms(obs, clip=False)
        return data

    def prepare_batches(self, data: dict) -> dict:
        """ Generates batches from independend paths. Batches all have the length of the
            configured sequence length.
            If path is not long enouth it gets padded and the padded values get masked out.
        Args:
            data (dict): dictionary holding information obtain from environment interactions
        Returns:
            dict: holding batched data with additional mask entry
        """
        path_slice_vals = data.pop("path_slice")
        # Re-slice with selected sequence length
        path_slice_new = []
        # Change slice sizes to sequence length
        l = max(self.seq_overlap,1)
        for s in path_slice_vals:
            for start_idx in range(int(s[0]), int(s[1]), l):
                path_slice_new.append([
                    int(start_idx),
                    int(min(start_idx + self.seq_len, s[1]))
                ])
        path_slice_vals = path_slice_new
        # Pad sequences
        l = max( s[1] - s[0] for s in path_slice_vals )
        data_batched = {
            k: [
                torch.nn.functional.pad(
                    data[k][s[0]:s[1]],
                    (*([0]*(len(data[k].size())*2 - 1)),l - s[1] + s[0]),
                    'constant',
                    0.0
                ) for s in path_slice_vals ] for k in data.keys()}
        # Generate Mask
        data_batched['mask'] = [
            torch.nn.functional.pad(
                torch.as_tensor( [*([1.0]*(s[1] - s[0]))], dtype=torch.float32 ),
                (0, l - s[1] + s[0]) ,
                'constant',
                0.0
            ) for s in path_slice_vals]

        return {k: torch.stack(v) for k, v in
                data_batched.items()}

    def prepare_memory(self) -> None:
        """ Prepare initial memory state of actor and critic
        """
        self.ac.pi.reset_states()
        self.ac.v.reset_states()

    def roll_out(self) -> None:
        """collect data and store to experience buffer."""
        o, ep_ret, ep_len = self.env.reset(), 0., 0
        self.ac.v.reset_states()
        self.ac.pi.reset_states()

        for t in range(self.local_steps_per_epoch):
            a, v, logp = self.ac.step(
                torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buf
            self.buf.store(obs=o, act=a, rew=r, val=v, logp=logp)
            self.logger.store(**{'Values/V': v})
            o = next_o

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = self.ac(
                        torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0.
                self.buf.finish_path(v)
                if terminal:  # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0., 0
                self.ac.v.reset_states()
                self.ac.pi.reset_states()

    def update_running_statistics(self, data) -> None:
        """ Update running statistics, e.g. observation standardization,
        or reward scaling. If MPI is activated: sync across all processes.
        """
        if self.use_standardized_obs:
            self.ac.obs_oms.update(data['obs'])

        # Apply Implement Reward scaling
        if self.use_reward_scaling:
            self.ac.ret_oms.update(data['discounted_ret'])

    def update(self) -> None:
        """Update value and policy networks. Note that the order doesn't matter.

        Returns
        -------
            None
        """
        raw_data = self.buf.get()
        # pre-process data: standardize observations, advantage estimation, etc.
        data = self.pre_process_data(raw_data)

        batched_data = self.prepare_batches(data)
        self.update_value_net(batched_data)
        self.update_policy_net(batched_data)

        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data) -> None:
        # initial RNN states
        self.prepare_memory()
        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        # get prob. distribution before updates: used to measure KL distance
        p_dist = self.ac.pi.dist(data['obs'])

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iterations):
            # initial RNN states
            self.prepare_memory()
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            if self.use_max_grad_norm:  # apply L2 norm
                torch.nn.utils.clip_grad_norm_(
                    self.ac.pi.parameters(),
                    self.max_grad_norm)
            # average grads across MPI processes
            mpi_tools.mpi_avg_grads(self.ac.pi.net)
            self.pi_optimizer.step()
            q_dist = self.ac.pi.dist(data['obs'])
            torch_kl = torch.distributions.kl.kl_divergence(
                p_dist, q_dist).mean().item()
            if self.use_kl_early_stopping:
                # average KL for consistent early stopping across processes
                if mpi_tools.mpi_avg(torch_kl) > self.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break

        # track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter': i + 1,
            'Values/Adv': data['adv'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio']
        })

    def update_value_net(self, data) -> None:
        no_batches = data["adv"].size()[0]
        mbs = int(max(no_batches / self.num_mini_batches, 1))

        if no_batches < self.num_mini_batches:
            loggers.warn(f"No. of batches ({no_batches}) " \
                         + f"smaller than no. mini batches ({self.num_mini_batches})")

        # initial RNN states
        self.prepare_memory()
        loss_v = self.compute_loss_v(data['obs'], data['target_v'],
                                     data['mask'])
        self.loss_v_before = loss_v.item()

        indices = np.arange(no_batches)
        val_losses = []
        start_indices = list(range(0, no_batches, mbs))
        for _ in range(self.train_v_iterations):
            random.shuffle(start_indices)
            for start_idx in range(self.num_mini_batches):
                start = start_indices[start_idx % len(start_indices)]
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                # initial RNN states
                self.prepare_memory()
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data['obs'][mb_indices],
                    ret=data['target_v'][mb_indices],
                    mask=data['mask'][mb_indices])
                val_losses.append(loss_v.item())
                loss_v.backward()
                # average grads across MPI processes
                mpi_tools.mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })


def get_alg(env_id, **kwargs) -> core.Algorithm:
    return IWPGAlgorithm(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='iwpg', env_id=env_id)
    defaults.update(**kwargs)
    alg = IWPGAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env
