r"""Script to run experiment: investigate influence of Hidden Neurons.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    22.11.2021
"""
import sys
import warnings

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
NUM_RUNS = 1

parameter_grid_dict = {
    "observation_history_size": [2, ],
    "ac_kwargs": [
        {"pi": {"activation": "relu", "hidden_sizes": [10, 10]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [11, 11]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [12, 12]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [13, 13]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [14, 14]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [15, 15]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [16, 16]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [17, 17]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [18, 18]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [19, 19]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [20, 20]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [21, 21]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [22, 22]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [23, 23]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [24, 24]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [25, 25]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [26, 26]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [27, 27]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [28, 28]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [29, 29]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [30, 30]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [31, 31]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [32, 32]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [33, 33]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [34, 34]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [35, 35]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [36, 36]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [37, 37]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [38, 38]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [39, 39]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [40, 40]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [41, 41]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [42, 42]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [43, 43]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [44, 44]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [45, 45]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [46, 46]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [47, 47]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [48, 48]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [49, 49]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        {"pi": {"activation": "relu", "hidden_sizes": [50, 50]}, "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},

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