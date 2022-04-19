""" Introduce an API which is similar to keras to train RL algorithms.

    Author:     Sven Gronauer
    Date:       19.05.2020
    Updated:    14.04.2022  - discarded multi-processing code snippets
"""
import torch
import os
from typing import Optional

from phoenix_drone_simulation.utils.loggers import setup_logger_kwargs
from phoenix_drone_simulation.utils import utils


class Model(object):

    def __init__(self,
                 alg: str,
                 env_id: str,
                 log_dir: str,
                 init_seed: int,
                 algorithm_kwargs: dict = {},
                 use_mpi: bool = False,
                 ) -> None:
        """ Class Constructor  """
        self.alg = alg
        self.env_id = env_id
        self.log_dir = log_dir
        self.init_seed = init_seed
        self.num_cores = 1  # set by compile()-method
        self.training = False
        self.compiled = False
        self.trained = False
        self.use_mpi = use_mpi

        self.default_kwargs = utils.get_defaults_kwargs(alg=alg,
                                                        env_id=env_id)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs['seed'] = init_seed
        self.kwargs.update(**algorithm_kwargs)
        self.logger_kwargs = None  # defined by compile (a specific seed might be passed)
        self.env_alg_path = os.path.join(self.env_id, self.alg)

        # assigned by class methods
        self.actor_critic = None
        self.env = None

    def _evaluate_model(self) -> None:
        from phoenix_drone_simulation.utils.evaluation import EnvironmentEvaluator
        evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
        evaluator.eval(env=self.env, ac=self.actor_critic, num_evaluations=128)
        # Close opened files to avoid number of open files overflow
        evaluator.close()

    def compile(self,
                num_cores=os.cpu_count(),
                exp_name: Optional[str] = None,
                **kwargs_update
                ) -> object:
        """Compile the model.

        Either use mpi for parallel computation or run N individual processes.

        Parameters
        ----------
        num_cores
        exp_name
        kwargs_update

        Returns
        -------

        """
        self.kwargs.update(kwargs_update)
        _seed = self.kwargs.get('seed', self.init_seed)

        if exp_name is not None:
            exp_name = os.path.join(self.env_alg_path, exp_name)
        else:
            exp_name = self.env_alg_path
        self.logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                 exp_name=exp_name,
                                                 seed=_seed)
        self.compiled = True
        self.num_cores = num_cores
        return self

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.env.render() if render else None
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            self.env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:
        self.actor_critic.eval()  # Set in evaluation mode before evaluation
        self._evaluate_model()
        self.actor_critic.train()  # switch back to train mode

    def fit(self, epochs=None, env=None) -> None:
        """ Train the model for a given number of epochs.

        Parameters
        ----------
        epochs: int
            Number of epoch to train. If None, use the standard setting from the
            defaults.py of the corresponding algorithm.
        env: gym.Env
            provide a custom environment for fitting the model, e.g. pass a
            virtual environment (based on NN approximation)

        Returns
        -------
        None

        """
        assert self.compiled, 'Call model.compile() before model.fit()'

        if epochs is None:
            epochs = self.kwargs.pop('epochs')
        else:
            self.kwargs.pop('epochs')  # pop to avoid double kwargs

        # fit() can also take a custom env, e.g. a virtual environment
        env_id = self.env_id if env is None else env

        learn_func = utils.get_learn_function(self.alg)
        ac, env = learn_func(
            env_id=env_id,
            logger_kwargs=self.logger_kwargs,
            epochs=epochs,
            **self.kwargs
        )
        self.actor_critic = ac
        self.env = env
        self.trained = True

    def play(self) -> None:
        """ Visualize model after training."""
        assert self.trained, 'Call model.fit() before model.play()'
        self.eval(episodes=5, render=True)

    def summary(self):
        """ print nice outputs to console."""
        raise NotImplementedError
