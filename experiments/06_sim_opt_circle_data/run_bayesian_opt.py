r"""Run simulation optimization (SimOpt) with Bayesian Optimization.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    29.10.2021
"""
import argparse
import numpy as np
import sys
import psutil
import time
import getpass
from skopt import Optimizer

# local imports
import phoenix_drone_simulation.utils.loggers as loggers
import phoenix_drone_simulation.utils.mpi_tools as mpi
from phoenix_drone_simulation.simopt.PyBullet import ObjectiveFunctionCircleTask


def run_bayesian_optimization(args):
    r"""Apply Bayesian Optimization to tune simulation parameters."""
    obj_func = ObjectiveFunctionCircleTask(seed=args.seed)

    param_names = ['thrust2weight', 'time_constant', 'dead_time']

    opt = Optimizer([# (0.030, 0.035),  # mass
                     (1.5, 2.5),  # thrust_to_weight_ratio
                     (0.01, 0.50),  # MOTOR_TIME_CONSTANT in [s]
                     (0.00, 0.05),  # LATENCY,  # in [s]
                     ])

    # set up logger
    user_name = getpass.getuser()
    logger_kwargs = loggers.setup_logger_kwargs(
        base_dir=f'/var/tmp/{user_name}',
        exp_name='simOpt/BayesianOpt',
        seed=args.seed
    )
    logger = loggers.EpochLogger(**logger_kwargs)

    # set up objective function
    def func(_x: np.ndarray) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return obj_func.evaluate(params=_x)

    # now run optimization
    mpi.mpi_print('*'*55)
    start_time = time.time()
    for epoch in range(500):
        logger.info(f'Start with epoch: {epoch}')
        suggested = np.array(opt.ask())
        mpi.broadcast(suggested)
        y = func(suggested)
        opt.tell(suggested.tolist(), y)
        res = opt.get_result()

        # do the logging and save to disk
        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('Loss', y)
        for i, value in enumerate(suggested):  # parameter values
            logger.log_tabular(f'Param/{param_names[i]}', value)
        for i, value in enumerate(res.x):  # parameter values
            logger.log_tabular(f'Best/{param_names[i]}', value)
        logger.log_tabular('Best/Loss', res.fun)
        logger.log_tabular('Time', int(time.time() - start_time))
        # now dump to disk...
        logger.dump_tabular()

    res = opt.get_result()
    loggers.info(f"Best parameters: \nJ_xx={res.x[0]} \nJ_yy= {res.x[1]} \nJ_zz= {res.x[2]}"
              f"\nT_s_T= {res.x[3]}\nK= {res.x[4]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    parser.add_argument('--cores', '-c', type=int, default=cores)
    # Random seed lies in range [0, 65535]
    parser.add_argument('--seed', default=int(time.time()) % 2**16)
    args = parser.parse_args()

    if mpi.mpi_fork(args.cores):
        # If number of cpus > 1: start parallel processes
        # Re-launches the current script with workers linked by MPI
        sys.exit()

    # mpi.setup_torch_for_mpi()
    run_bayesian_optimization(args)
