import os
import sys
import unittest
import gym
import pybullet_envs  # noqa
import time
import numpy as np
import psutil

# local imports
import phoenix_drone_simulation.utils.mpi_tools as mpi
import phoenix_drone_simulation.utils.utils as U
from phoenix_drone_simulation.utils.loggers import setup_logger_kwargs


IMPLEMENTED_ALGS = ['iwpg', 'npg', 'trpo', 'ppo' ]

class TestAlgorithms(unittest.TestCase):

    @staticmethod
    def check_alg(alg_name, env_id, cores):
        """" Run one epoch update with algorithm."""
        print(f'Run {alg_name}.')
        defaults = U.get_defaults_kwargs(alg=alg_name, env_id=env_id)
        defaults['epochs'] = 1
        defaults['num_mini_batches'] = 4
        defaults['steps_per_epoch'] = 1000 * cores
        defaults['verbose'] = False

        defaults['logger_kwargs'] = setup_logger_kwargs(
            exp_name='unittest',
            seed=0,
            base_dir='/var/tmp/',
            datestamp=True,
            level=0,
            use_tensor_board=True,
            verbose=False)
        alg = U.get_alg_class(alg_name, env_id, **defaults)
        # sanity check of argument passing
        assert alg.alg == alg_name, f'Expected {alg_name} but got {alg.alg}'
        # return learn_fn(env_id, **defaults)
        ac, env = alg.learn()

        return ac, env

    # def test_algorithms_single_core(self):
    #     """ Run all the specified algorithms with only a single CPU core."""
    #     for alg in IMPLEMENTED_ALGS:
    #         ac, env = self.check_alg(alg, 'DroneCircleBulletEnv-v0', cores=1)
    #         self.assertTrue(isinstance(env, gym.Env))


    def test_MPI_version_of_algorithms(self):
        """ Run all the specified algorithms with MPI."""
        # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
        physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
        assert physical_cores > 1, \
            f'Cannot test MPI when only one core is available.'
        if mpi.mpi_fork(
                n=physical_cores):  # forks the current script and use MPI
            return  # use return instead of sys.exit() to exit test with 'OK'...
        is_root = mpi.proc_id() == 0
        for alg in IMPLEMENTED_ALGS:
            try:
                mpi.mpi_print(f'Run MPI with {alg.upper()}')
                ac, env = self.check_alg(alg, 'DroneCircleBulletEnv-v0',
                                         cores=physical_cores)
                self.assertTrue(isinstance(env, gym.Env))
            except NotImplementedError:
                print('No MPI yet supported...') if is_root else None
        else:
            # sleep one sec to finish all console prints...
            time.sleep(1)

if __name__ == '__main__':
    unittest.main()
