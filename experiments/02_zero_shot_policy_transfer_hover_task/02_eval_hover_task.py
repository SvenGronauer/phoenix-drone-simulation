r"""Evaluate Hover task performance of KW 36 / 2021.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    11.09.2021
"""
import os
from scipy import signal
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch as th
import torch.nn as nn
import glob

import phoenix_drone_simulation  # noqa
from phoenix_drone_simulation.envs.utils import rad2deg

LOG_FREQ = 100  # [HZ]


def get_frequency_from(df: pd.DataFrame, col: str, debug=False) -> float:
    N = df.shape[0]
    vals = df[col].to_numpy(dtype=np.float64)
    yf = scipy.fft.fft(vals)
    xf = scipy.fft.fftfreq(N, 1 / LOG_FREQ)
    yf_idx = np.argmax(yf)
    if yf_idx == 0:  # filter 0 Hz for PWM signals...
        yf_idx = np.argmax(yf[1:])
        a = 5
    freq = np.abs(xf[yf_idx])
    if debug:
        print(f'{col}\t freq: {np.abs(xf[yf_idx])}')
        plt.plot(xf, yf)
        plt.show()
    return freq


def get_infos_from(dir_name: str, csv_fnp: str) -> dict:
    df = pd.read_csv(csv_fnp)
    infos = {}
    infos['Name'] = dir_name
    # === Oscillation Frequencies
    infos['freq_roll'] = get_frequency_from(df, 'roll')
    infos['freq_roll_dot'] = get_frequency_from(df, 'roll_dot')
    infos['freq_pitch'] = get_frequency_from(df, 'pitch')
    infos['freq_pitch_dot'] = get_frequency_from(df, 'pitch_dot')

    # === Sum of angle errors
    infos['mse_roll'] = np.mean(rad2deg(df['roll'].to_numpy(dtype=np.float))**2)
    infos['mse_pitch'] = np.mean(rad2deg(df['pitch'].to_numpy(dtype=np.float))**2)

    # flight duration
    infos['flight_time'] = len(df['time']) / LOG_FREQ

    # motor frequencies
    infos['freq_mot0'] = get_frequency_from(df, 'n0', debug=False)
    infos['freq_mot1'] = get_frequency_from(df, 'n1')
    infos['freq_mot2'] = get_frequency_from(df, 'n2')
    infos['freq_mot3'] = get_frequency_from(df, 'n3')
    return infos


def eval_all_csv_in_path(path):
    df = None

    for dir_name in os.listdir(path):
        if dir == '.DS_Store':
            continue
        current_dir = os.path.join(path, dir_name)
        # walk recursively trough basedir
        for root, dirs, files in os.walk(current_dir):
            config_json_in_dir = False
            metrics_json_in_dir = False
            for file in files:
                if file.endswith(".csv"):
                    csv_fnp = os.path.join(root, file)
                    infos = get_infos_from(dir_name, csv_fnp)
                    # append to df
                    if df is None:
                        df = pd.DataFrame(columns=infos.keys())
                    df = df.append(infos, ignore_index=True)

        csv_file_paths = glob.glob(os.path.join(path, dir_name, '*.csv'))

    df.sort_values(by=['Name'], inplace=True)
    return df


def main():
    this_fp = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    path_deadtime_10ms = os.path.join(this_fp, '2021_KW_36_logs/exp_02_motor_time_constant_10ms_dead_time')
    print(f'Data path: {os.path.abspath(path_deadtime_10ms)}')

    print('*'*55)
    print(path_deadtime_10ms)
    df = eval_all_csv_in_path(path_deadtime_10ms)
    print(df)


if __name__ == '__main__':
    main()
