r"""Run simulation optimization (SimOpt) with a finite-difference method based
on Adam optimizer.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    28.10.2021
"""
import argparse
import numpy as np
from scipy import optimize
import sys
import psutil
import time
import getpass
import torch as th
import torch.optim

# local imports
import phoenix_drone_simulation.utils.loggers as loggers
import phoenix_drone_simulation.utils.mpi_tools as mpi
from phoenix_drone_simulation.simopt.pybullet import ObjectiveFunctionCircleTask


def optimize_with_Adam(args):
    r"""Apply Adams' Stochastic Gradient Descend based on finite-differences."""
    obj_func = ObjectiveFunctionCircleTask(seed=args.seed)

    # set up logger
    user_name = getpass.getuser()
    logger_kwargs = loggers.setup_logger_kwargs(
        base_dir=f'/var/tmp/{user_name}',
        exp_name='simOpt/Adam',
        seed=args.seed
    )
    logger = loggers.EpochLogger(**logger_kwargs)

    # start with motor_time_constant = 0 and latency = 0
    x0 = np.array([2.25, 0., 0.])
    mpi.broadcast(x0)
    mpi.mpi_print('x0:', x0)

    # define parameter ranges
    lower = th.tensor([1.5, 0.0, 0.0])
    higher = th.tensor([2.5, 0.50, 0.05])

    # load adam Opt
    params = th.tensor(x0, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=1e-3)

    # set up objective function
    def func(_x: np.ndarray) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return obj_func.evaluate(params=_x)

    # set up finite-difference approximation function
    eps = 1000 * np.sqrt(np.finfo(float).eps)
    mpi.mpi_print('eps:', eps)
    # absolute step size used for numerical approximation of the Jacobian via
    # forward differences:
    epsilon_vector = [np.sqrt(x) * eps for x in x0]
    epsilon_vector[0] = 0.001
    epsilon_vector[1] = 0.001
    epsilon_vector[2] = 0.005  # set equal to simulation time step (1/sim_freq)
    mpi.mpi_print('epsilon_vector:', epsilon_vector)

    def jac_func(_x: np.ndarray):
        r"""Calculates Jacobian based on finite-difference approximation."""
        # return optimize.approx_derivative(
        #     func, _x, method='3-point', abs_step=epsilon_vector, f0=None
        # )
        return optimize.approx_fprime(_x, func, epsilon_vector)

    # now run optimization
    mpi.mpi_print('*'*55)
    x = x0.copy()

    start_time = time.time()
    for epoch in range(500):
        logger.info(f'Start with epoch: {epoch}')
        grad = mpi.mpi_avg(jac_func(params.detach().numpy()))

        # set gradient to Torch parameters:
        optimizer.zero_grad()
        params.backward(th.tensor(grad))
        optimizer.step()

        # Clip parameters into desired range
        # Note: clamp method accepts tensors for Torch versions >=1.10
        params.data = th.max(th.min(params.data, higher), lower)

        # do the logging and save to disk
        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('Loss', func(params.detach().numpy()))
        for i, value in enumerate(params.detach().numpy()):  # parameter values
            logger.log_tabular(f'Parameters/{i}', value)
        for i, value_grad in enumerate(grad):  # gradient information
            logger.log_tabular(f'Gradient/{i}', value_grad)

        logger.log_tabular('Time', int(time.time() - start_time))
        # now dump to disk...
        logger.dump_tabular()


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

    mpi.setup_torch_for_mpi()
    optimize_with_Adam(args)
