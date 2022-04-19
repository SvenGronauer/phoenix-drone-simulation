r"""Search for NN architectures for real-world Drone Hover task.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    10-08-2021
"""
import sys
import warnings
import copy
import sys
import numpy as np
import psutil
from phoenix_drone_simulation.train import get_training_command_line_args, run_training


def get_grid() -> list:

    return [
        # 2021/KW33:
        (50, 50, 'relu'),
        (50, 50, 'tanh'),
        # (50, 30, 20, 'relu'),
        # (50, 30, 20, 'tanh'),
        # (32, 32, 'relu'),
        # (32, 32, 'tanh'),
        # (16, 16, 'relu'),
        # (16, 16, 'tanh'),
    ]

    # return [
    #     # 2021/KW32: upper computational limit on drone is ~4000 Parameters
    #     (50, 50, 'relu'),
    #     (50, 50, 'tanh'),
    #     (40, 40, 'relu'),
    #     (40, 40, 'tanh'),
    #     (50, 30, 20, 'relu'),
    #     (50, 30, 20, 'tanh'),
    #     (40, 40, 40, 'relu'),
    #     (40, 40, 40, 'tanh'),
    #     (30, 30, 30, 'relu'),
    #     (30, 30, 30, 'tanh'),
    #     (32, 32, 'relu'),
    #     (32, 32, 'tanh'),
    #     (16, 16, 'relu'),
    #     (16, 16, 'tanh'),
    #
    #     (32, 32, 32, 32, 'relu'),  # = 3736 Parameters
    #     (32, 32, 32, 32, 'tanh'),  # = 3736 Parameters
    #     (25, 25, 25, 25, 'relu'),  # = 2400 Parameters
    #     (25, 25, 25, 25, 'tanh'),  # = 2400 Parameters
    # ]



if __name__ == '__main__':
    physical_cores = 2**int(np.log2(psutil.cpu_count(logical=False)))
    start = sys.argv
    adds = [
        '--env',  'DroneHoverBulletEnv-v0',
        '--alg',  'trpo',
        '-c', str(physical_cores),
        '--epochs', '500'
    ]
    eaz = sys.argv
    sys.argv += adds
    search_grid = get_grid()

    default_args, default_unparsed_args = get_training_command_line_args()

    for config in search_grid:
        args = copy.deepcopy(default_args)
        args.pi = config
        exp_name = '_'.join([str(s) for s in config])
        run_training(
            args, unparsed_args=default_unparsed_args, exp_name=exp_name)
