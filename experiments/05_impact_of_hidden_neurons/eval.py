r"""Script to evaluate experiment: influence of NN hidden neuron size.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    23.11.2021
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# local imports
import phoenix_drone_simulation  # noqa
import phoenix_drone_simulation.utils.export as export

def get_number_neurons_and_mean_episode_return(
        path: str
) -> Tuple[int, float]:
    fnp = os.path.join(path, 'returns.csv')
    ep_returns = export.get_file_contents(fnp)

    conf = export.get_file_contents(os.path.join(path, 'config.json'))
    hidden_sizes = conf['ac_kwargs']['pi']['hidden_sizes']
    number_neurons = hidden_sizes[0]  # looks like: [64, 64]
    epRet = np.mean(ep_returns)
    # print(f'Mean ret: {epRet:0.3f} from {fnp}')
    return number_neurons, epRet

def main():
    current_file_dir = os.getcwd()
    data_path = os.path.join(current_file_dir, 'data/')
    print(f'Data path: {data_path}')

    # === Find all directories (expecting 75) that contain a `config.json` file
    i = 0
    run_directories = []
    for run_dir, dirs, files in os.walk(data_path):  # walk recursively trough data
        for file in files:
            if file == "config.json":
                i += 1
                # print(f'File {i} at: {file} dir: {run_dir}')
                run_directories.append(run_dir)
    print(f'Found: {i} directories')

    neurons = []
    epRets = []
    for dr in sorted(run_directories):
        print(dr)
        nn, ret = get_number_neurons_and_mean_episode_return(dr)
        neurons.append(nn)
        epRets.append(ret)

    plt.scatter(neurons, epRets)
    plt.xlabel('Number of Hidden Neurons')
    plt.ylabel('Mean Episode Return')
    plt.title('DroneCircleBulletEnv-v0')
    plt.show()


if __name__ == '__main__':
    main()
