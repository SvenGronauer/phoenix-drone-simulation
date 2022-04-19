r"""Script to run Zero-shot transfer to CrazyFlie.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    06.09.2021
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
from phoenix_drone_simulation.benchmark import Benchmark

parameter_grid_dict = {
    "penalty_action": [0.1, ],  #  [0.01, 0.1],
    "penalty_spin": [0.1, ],  #  [0.01, 0.1],
    "domain_randomization": [0.01, ],  # [0.01, 0.1],
    "motor_time_constant": [0.050, 0.100, 0.150, 0.200],

    # NOTE: these values might be used for Take-Off experiemnts:
    # +    "penalty_action": [0., ],  #  [0.01, 0.1],
    # +    "penalty_spin": [0., ],  #  [0.01, 0.1],
    # +    "domain_randomization": [0.001, 0.05, 0.1],  # [0.01, 0.1],
    # +    "motor_time_constant": [0.150, ],
}

alg_setup = {
    'ppo': parameter_grid_dict,
}


def get_arguments():
    r"""Fetches command line arguments from sys.argv."""
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2**16
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cores', '-c', type=int, default=physical_cores,
                        help='Number of parallel processes generated.')
    parser.add_argument('--runs', '-r', type=int, default=2,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/ga87zej',
                        help='Define a custom directory for logging.')
    parser.add_argument('--seed', type=int, default=random_seed,
                        help='Define the initial seed.')
    args = parser.parse_args()
    return args


def main(args):

    env_specific_kwargs = {
        'DroneTakeOffBulletEnv-v0': {
            "ac_kwargs": {
                "pi":	{"activation":	"relu", "hidden_sizes":	[50, 50]},
                "val":	{"activation":	"tanh", "hidden_sizes":	[64, 64]}},
            'epochs': 1, 'steps_per_epoch': 32000},
    }
    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.cores,
        num_runs=args.runs,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        init_seed=args.seed,
    )
    bench.run()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
