r"""Script to evaluate experiment: influence of state-action history.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    15.12.2021
"""
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List, Union

# local imports
import phoenix_drone_simulation  # noqa
import phoenix_drone_simulation.utils.export as export
import phoenix_drone_simulation.utils.utils as utils

CONTROL_MODES = ['PWM', 'AttitudeRate', 'Attitude']
PARAMETERS = ['PWM', 'AttitudeRate', 'Attitude']

LOG_FREQUENCY = 100
MAX_FLIGHT_TIME = 20.0  # in seconds

# global data dictionary:
DATA = {}


# Each neural network is flown three times (m1, m2, m3)
NetworkMeasurement = namedtuple(
    'NetworkMeasurement',
    'control_mode latency motor_time_constant seeds data'
)


def get_setting_from_measurements(measurements: List[NetworkMeasurement],
                                   latency: float,
                                   mtc: float
                                   ) -> Union[NetworkMeasurement, None]:
    for m in measurements:
        if m.motor_time_constant == mtc and m.latency == latency:
            return m
    return None


def get_setting_with_seed(measurements: List[NetworkMeasurement],
                          seed: int
                          ) -> Union[NetworkMeasurement, None]:
    for m in measurements:
        if seed in m.seeds:
            return m
    return None


def initialize_network_measurements(control_mode: str,
                                    data_path: str
                                    ) -> List[NetworkMeasurement]:
    r"""Find all directories that contain a `env_config.json` file."""
    i = 0
    run_directories = []
    measurements = []
    for run_dir, dirs, files in os.walk(data_path):  # walk recursively trough data
        for file in files:
            if file == "env_config.json":
                i += 1
                # print(f'File {i} at: {file} dir: {run_dir}')
                run_directories.append(run_dir)
                env_config = utils.get_file_contents(os.path.join(run_dir, file))
                alg_config = utils.get_file_contents(os.path.join(run_dir, 'config.json'))
                assert env_config['agent_params']['control_mode'] == control_mode
                mtc = env_config['agent_params']['motor_time_constant']
                latency = env_config['agent_params']['latency']
                seed = alg_config['seed']

                m = get_setting_from_measurements(measurements, latency=latency, mtc=mtc)
                if m is not None:
                    m.seeds.append(seed)
                else:
                    measurement = NetworkMeasurement(
                        control_mode=control_mode,
                        latency=latency,
                        motor_time_constant=mtc,
                        seeds=[seed, ],
                        data=[]
                    )
                    measurements.append(measurement)

    print(f'Found: {i} configs.')
    for m in measurements:
        print(m)
    print('='*55)
    return measurements


def get_flight_time_from_csv(path, file_name):
    fnp = os.path.join(path, file_name)
    data = pd.read_csv(fnp)
    flight_duration = len(data) / LOG_FREQUENCY
    flight_time = min(flight_duration, MAX_FLIGHT_TIME)
    return flight_time


def load_measurements(measurements: List[NetworkMeasurement],
                      log_path: str):
    for path, dirs, files in os.walk(log_path):  # walk recursively trough data
        for file_name in files:
            if file_name.endswith('.csv'):
                control_seed_number = path.split(os.sep)[-2]  # e.g. PWM_seed_00016
                seed = int(control_seed_number.split('_')[-1])
                # print(f'path: {path} file_name: {file_name} seed: {seed}')
                s = get_setting_with_seed(measurements, seed)
                assert s is not None
                flight_time = get_flight_time_from_csv(path, file_name)
                s.data.append(flight_time)


def fill_data_dict():

    for cm in CONTROL_MODES:
        current_file_dir = os.getcwd()
        data_path = os.path.join(current_file_dir, 'checkpoints', cm)
        DATA[cm] = initialize_network_measurements(cm, data_path=data_path)
        log_path = os.path.join(current_file_dir, 'logs', cm)
        load_measurements(DATA[cm], log_path)


def plot_data_dict():
    # for latency in [0.01, 0.015, 0.020]:

    for control_mode in DATA.keys():
        for latency in [0.0, 0.015, 0.020]:
            ys = []
            xs = [0.04, 0.08, 0.12]
            for mtc in xs:
                m = get_setting_from_measurements(
                    DATA[control_mode], mtc=mtc, latency=latency)
                assert m is not None
                ys.append(np.mean(m.data))
            plt.plot(xs, ys, label=f'Latency={latency} [{control_mode}]')

    plt.ylabel('Mean Flight Time [s]')
    plt.xlabel('Motor Time constant [s]')
    plt.title('PWM vs Attitude Control Structure')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # args, unparsed_args = get_training_command_line_args(
    #     alg=ALG, env=ENV, num_runs=NUM_RUNS)
    args = None
    fill_data_dict()
    print('*'*55)
    for cm in CONTROL_MODES:
        for m in DATA[cm]:
            print(m)
    plot_data_dict()
