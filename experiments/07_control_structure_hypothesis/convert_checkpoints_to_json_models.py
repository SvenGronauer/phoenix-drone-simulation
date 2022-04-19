r"""Script to evaluate experiment: influence of state-action history.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    14.12.2021
"""
import os
import numpy as np
import shutil

# local imports
import phoenix_drone_simulation  # noqa
import phoenix_drone_simulation.utils.export as export


def main(control_mode: str):
    assert control_mode in ['PWM', 'AttitudeRate', 'Attitude']
    current_file_dir = os.getcwd()
    data_path = os.path.join(current_file_dir, 'checkpoints', control_mode)
    print(f'Data path: {data_path}')

    # === Find all directories that contain a `config.json` file
    i = 0
    run_directories = []
    for run_dir, dirs, files in os.walk(data_path):  # walk recursively trough data
        for file in files:
            if file == "config.json":
                i += 1
                # print(f'File {i} at: {file} dir: {run_dir}')
                run_directories.append(run_dir)
    print(f'Found: {i} directories')

    # === Create model.json files
    for run_dir in list(sorted(run_directories)):
        print(f'===\nCurrent Dir: {run_dir}')
        ac, env = utils.load_actor_critic_and_env_from_disk(run_dir)
        seed_string = control_mode + '_' + run_dir.split(os.sep)[-1]
        file_name = seed_string + '_model.json'  # e.g. PWM_seed_00XX_model.json
        file_path = f'/var/tmp/models/{control_mode}'
        os.makedirs(file_path, exist_ok=True)
        export.convert_actor_critic_to_json(
            actor_critic=ac, file_path=file_path, file_name=file_name
        )


if __name__ == '__main__':
    main('Attitude')
