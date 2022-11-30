import argparse
import numpy as np
import psutil

import phoenix_drone_simulation  # noqa
from phoenix_drone_simulation.benchmark import Benchmark
"""

        ac_kwargs={
            'pi': {
                'hidden_sizes': [20, 20],
                'layers': ['lstm', 'tanh', ],
                # 'activation': 'identity',
                # 'hidden_sizes': [18, 18],
                # # 'hiddens': [20, "lstm", 20, "tanh", 20, "tanh"],
                # 'layer': 'lstm'
            },
            'val': {
                'hidden_sizes': [64, 64],
                'layers': ['lstm', 'tanh', 'identity'],
                # 'activation': 'identity',
                # 'hidden_sizes': [64, 64],
                # 'layer': 'lstm'
            }
"""

hyper_param_search = [
    {'pi': {'hidden_sizes': [10, 10], 'layers': ['lstm', 'tanh', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [20, 20], 'layers': ['lstm', 'tanh', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [30, 30], 'layers': ['lstm', 'tanh', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [40, 40], 'layers': ['lstm', 'tanh', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [50, 50], 'layers': ['lstm', 'tanh', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    
    {'pi': {'hidden_sizes': [10, 10], 'layers': ['tanh', 'lstm', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [20, 20], 'layers': ['tanh', 'lstm', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [30, 30], 'layers': ['tanh', 'lstm', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [40, 40], 'layers': ['tanh', 'lstm', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},
    {'pi': {'hidden_sizes': [50, 50], 'layers': ['tanh', 'lstm', ]}, 'val': {'hidden_sizes': [64, 64], 'layers': ['lstm', 'tanh', ]}},

    
]

alg_setup = {
    'ppo': {'ac_kwargs': hyper_param_search}
}

env_specific_kwargs = {
    'DroneCircleBulletEnv-v0': {'epochs': 1000, 'steps_per_epoch': 32000},
}


def argument_parser():
    n_cpus = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=3,
                        help='Number of indipendent seeds per experiment.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/ga87zej',
                        help='Define a custom directory for logging.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Define the initial seed.')
    args = parser.parse_args()
    return args


def main(args):

    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.num_cores,
        num_runs=args.num_runs,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        init_seed=args.seed,
    )
    bench.run()


if __name__ == '__main__':
    args = argument_parser()
    main(args)
