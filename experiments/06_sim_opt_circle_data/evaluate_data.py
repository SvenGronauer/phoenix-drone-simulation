r"""Evaluate simulation optimization (SimOpt) with Bayesian Optimization.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    27.12.2021
"""
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


MINI_TRAJECTORY_LENGTHS = [10, 20, 30, 40, 50]

current_file_dir = os.getcwd()
data_path = os.path.join(current_file_dir, 'data')


def get_results_from_csv_files(T: int
                               ) -> Tuple[List[float], List[float], List[float]]:
    directory_path = os.path.join(data_path, f'T_{T}')

    thrust_to_weight_ratios = []
    motor_time_constants = []
    latencies = []

    # walk recursively trough data:
    for run_dir, dirs, files in os.walk(directory_path):
        for file in files:
            if file == "progress.csv":
                file_name_path = os.path.join(run_dir, file)
                df = pd.read_csv(file_name_path)

                # Read the following columns from CSV:
                # Best/thrust2weight	Best/time_constant	Best/dead_time
                thrust_to_weight_ratios.append(df['Best/thrust2weight'].values[-1])
                motor_time_constants.append(df['Best/time_constant'].values[-1])
                latencies.append(df['Best/dead_time'].values[-1])
    return thrust_to_weight_ratios, motor_time_constants, latencies


def print_mean_max_min(name: str, values: np.array):
    print(f'{name}:\t'
          f'Mean: {np.mean(values):0.3f}'
          f'+/-{np.std(values):0.4f} (stddev)\t'
          f'Max: {np.max(values):0.4f}\t'
          f'Min{np.min(values):0.4f}')


def evaluate_sim_opt_results() -> None:
    for T in MINI_TRAJECTORY_LENGTHS:
        print('*'*55, f'\nT={T}\n')
        t2w, mtc, latencies = get_results_from_csv_files(T=T)
        print_mean_max_min(name="Thrust-to-weight", values=t2w)
        print_mean_max_min(name="MTC", values=mtc)
        print_mean_max_min(name="Latency", values=latencies)


if __name__ == '__main__':
    evaluate_sim_opt_results()
