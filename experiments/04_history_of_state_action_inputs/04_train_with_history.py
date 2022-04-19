r"""Script to run experiment: investigate influence of state-action history.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    05.10.2021
"""
import sys
import warnings

try:
    import phoenix_drone_simulation
except ImportError as e:
    warnings.warn(f'Could not load Algorithm repository of Sven :-(')
    sys.exit()

import argparse
import time
import psutil
import numpy as np

# local imports
import phoenix_drone_simulation
from phoenix_drone_simulation.train import get_training_command_line_args
from phoenix_drone_simulation.benchmark import Benchmark

# ------------------------------------------------------------------------------
#   Adjustable parameters for the experiment
# ------------------------------------------------------------------------------

ALG = 'ppo'
ENV = 'DroneCircleBulletEnv-v0'
NUM_RUNS = 5

parameter_grid_dict = {
    "observation_history_size": [1, 2, 4, 6, 8],
    "ac_kwargs": [
        {"pi": {"activation": "relu", "hidden_sizes": [32, 32]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [48, 48]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [64, 64]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
    ]
}


# ------------------------------------------------------------------------------
#   Fixed parameters of the experiment
# ------------------------------------------------------------------------------
env_specific_kwargs = {
    ENV: {
        'epochs': 1000,
        'steps_per_epoch': 64000,
        'domain_randomization': 0.01,
        'observation_noise': 1,  # sensor noise enabled when > 0
        'motor_time_constant': 0.150,  # [s]
        'motor_thrust_noise': 0.05,  # noise in % added to thrusts
        'penalty_spin': 0.001,
    },
}


def main(args):
    alg_setup = {
        ALG: parameter_grid_dict,
    }
    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.cores,
        num_runs=NUM_RUNS,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        init_seed=0,  # start with seed 0 and then count up
    )
    bench.run()


if __name__ == '__main__':
    args, unparsed_args = get_training_command_line_args(
        alg=ALG, env=ENV)
    main(args)
