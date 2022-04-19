r"""Script to evaluate action distribution of real-world flights.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    27.12.2021
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


def get_data(log_path):
    i = 0
    for run_dir, dirs, files in os.walk(log_path):  # walk recursively trough data
        for file in files:
            if file.endswith('.csv'):
                i += 1



def main():
    current_file_dir = os.getcwd()
    log_path = os.path.join(current_file_dir, 'logs')
    control_mode = 'PWM'


if __name__ == '__main__':
    main()
