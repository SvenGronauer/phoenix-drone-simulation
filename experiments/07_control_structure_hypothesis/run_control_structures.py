r"""Script to run experiment: Control Structure Hypothesis.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    13.12.2021
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
NUM_RUNS = 3

parameter_grid_dict = {
    "control_mode": ['PWM', 'AttitudeRate', 'Attitude'],
    "motor_time_constant": [0.040, 0.060, 0.080, 0.100, 0.120],  # [s]
    "latency": [0.010, 0.015, 0.020],  # [s]
}

# ------------------------------------------------------------------------------
#   Fixed parameters of the experiment
# ------------------------------------------------------------------------------
env_specific_kwargs = {
    ENV: {
        'ac_kwargs': {"pi": {"activation": "relu", "hidden_sizes": [50, 50]},
                      "val": {"activation": "tanh", "hidden_sizes": [64, 64]}},
        'epochs': 500,
        'steps_per_epoch': 64000,
        'domain_randomization': 0.10,
        'observation_noise': 1,  # sensor noise enabled when > 0
        'observation_history_size': 2,  # use last two states and inputs
        'motor_thrust_noise': 0.05,  # noise in % added to thrusts
    },
}


def main(args):
    alg_setup = {
        ALG: parameter_grid_dict,
    }
    control_modes = {  # keys: control mode, values: aggregate_phy_steps
        'PWM': 2,
        'AttitudeRate': 4,
        'Attitude': 8
    }
    i = 0
    for (control_mode, aggregate_phy_steps) in control_modes.items():
        env_specific_kwargs[ENV]['control_mode'] = control_mode
        env_specific_kwargs[ENV]['aggregate_phy_steps'] = aggregate_phy_steps
        bench = Benchmark(
            alg_setup,
            env_ids=list(env_specific_kwargs.keys()),
            log_dir=args.log_dir,
            num_cores=args.cores,
            num_runs=NUM_RUNS,
            env_specific_kwargs=env_specific_kwargs,
            use_mpi=True,
            init_seed=i,  # start with seed 0 and then count up
        )
        bench.run()


if __name__ == '__main__':
    args, unparsed_args = get_training_command_line_args(
        alg=ALG, env=ENV)
    main(args)