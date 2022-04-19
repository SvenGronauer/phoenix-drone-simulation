from phoenix_drone_simulation.simopt.pybullet import ObjectiveFunctionCircleTask
import numpy as np
import sys
from deap import base, creator
from deap import cma
import math
import pickle
import time
import argparse
import phoenix_drone_simulation.utils.mpi_tools as mpi


def optimize_with_CMAES(sigma=0.1, nb_generations=100, nb_individuals=10):
    obj_func = ObjectiveFunctionCircleTask()

    # sample initial guess about drone's system parameters
    #scaling_factors = np.array([2.25,0,0]) # obj_func.sample()
    # NOTE That having 0s in the scaling factors at the beginning results in problems (obiously) that's why I start now in the middle of the parameter space
    scaling_factors = (obj_func.parameter_space.high + obj_func.parameter_space.low) / 2
    x0 = np.ones(shape=(len(scaling_factors),))
    mpi.mpi_print('x0:', x0 * scaling_factors)

    def convert_individual(_x):
        return _x * scaling_factors

    # set up objective function
    def func(_x) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        converted_individual = convert_individual(_x)
        if obj_func.parameter_space.contains(converted_individual):
            res = obj_func.evaluate(params=converted_individual)
            if not math.isnan(res):
                return res
        return 1e9

    # Setup the CMA ES search
    creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
    creator.create("Individual",np.ndarray,fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate",func)

    strategy = cma.Strategy(centroid=x0,sigma=sigma,lambda_=nb_individuals) # centroid: An iterable object that indicates where to start the evolution, sigma: The initial standard deviation of the distribution
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Keep track of the best individual:
    best_individuals = [[convert_individual(x0), 1e6]]

    for gen in range(nb_generations):
        gen_start = time.time()

        pop = toolbox.generate()

        fitnesses = list(map(toolbox.evaluate,pop))

        for ind, fit in zip(pop,fitnesses):
            ind.fitness.values = (fit,)

        best_individuals.append([convert_individual(pop[np.argmin(fitnesses)]), np.min(fitnesses)])

        mpi.mpi_print(f"Fitness of the best individual in Generation {gen}: {np.min(fitnesses)} ({str(best_individuals[-1])})")
        mpi.mpi_print(f"Generation took {(time.time() - gen_start)} seconds")

        toolbox.update(pop)

        pickle.dump(best_individuals, open("best_individuals.p", "wb"))

    mpi.mpi_print("Best individual: " + str(best_individuals[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', '-c', type=int, default=1)
    args = parser.parse_args()

    if mpi.mpi_fork(args.cores, use_number_of_threads=True):
        # If number of cpus > 1: start parallel processes
        # Re-launches the current script with workers linked by MPI
        sys.exit()

    mpi.setup_torch_for_mpi()
    
    optimize_with_CMAES(sigma=0.01, nb_generations=10000, nb_individuals=100000)
