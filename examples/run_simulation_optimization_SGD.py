import argparse
from phoenix_drone_simulation.simopt.pybullet import ObjectiveFunctionCircleTask
import phoenix_drone_simulation.utils.mpi_tools as mpi
import numpy as np
from scipy import optimize
import sys
import psutil


def example_evaluate_fitness_score(*args, **kwargs):
    obj_func = ObjectiveFunctionCircleTask()

    # sample initial guess about drone's system parameters
    x = obj_func.sample()

    def func(_x) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return obj_func.evaluate(params=_x)

    print(f'Fitness score = {func(x)}')


def optimize_with_SGD():
    """Apply Stochastic Gradient Descend (SGD) based on finite-differences."""
    obj_func = ObjectiveFunctionCircleTask()

    # sample initial guess about drone's system parameters
    x0 = mpi.mpi_avg(obj_func.sample())
    mpi.mpi_print('x0:', x0)

    # set up objective function
    def func(_x) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return obj_func.evaluate(params=_x)

    print(f'f(x0)={func(x0)}')

    # set up finite-difference approximation function
    eps = 1000 * np.sqrt(np.finfo(float).eps)
    mpi.mpi_print('eps:', eps)
    # absolute step size used for numerical approximation of the Jacobian via
    # forward differences:
    epsilon_vector = [np.sqrt(x) * eps for x in x0]
    mpi.mpi_print('epsilon_vector:', epsilon_vector)

    def jac_func(_x):
        r"""Calculates Jacobian based on finite-difference approximation."""
        return optimize.approx_fprime(_x, func, epsilon_vector)

    jac = jac_func(x0)
    # mpi.mpi_print('Jac:', jac)
    # mpi.mpi_print('norm of Jac:', np.linalg.norm(jac))
    # mpi.mpi_print('=' * 55)
    # mpi.mpi_print('Start to optimize..')

    options = {
        'gtol': 1e3,  # Gradient norm must be less than `gtol` before successful termination.
        'disp': mpi.is_root_process(),
        'maxiter': 20  # Maximum number of iterations to perform.
    }

    # now run optimization
    mpi.mpi_print('*'*55)
    mpi.mpi_print(f'Init f(x0): {func(x0)}')
    x = x0.copy()
    lr = 1e-8
    for epoch in range(500):
        grad = mpi.mpi_avg(jac_func(x0))
        x = x - lr * grad  # * x
        mpi.mpi_print(f'x{epoch+1}: {x} via grad: {grad}')
        mpi.mpi_print(f'func(x{epoch+1}) = {func(x)}')
    # res = optimize.minimize(
    #     func, x0, method='SGD', jac=jac_func, options=options
    # )
    mpi.mpi_print(f'Final result: f(x)={func(x)} \tx={x}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    parser.add_argument('--cores', '-c', type=int, default=cores)
    args = parser.parse_args()

    if mpi.mpi_fork(args.cores):
        # If number of cpus > 1: start parallel processes
        # Re-launches the current script with workers linked by MPI
        sys.exit()

    mpi.setup_torch_for_mpi()
    # example_evaluate_fitness_score()
    optimize_with_SGD()
