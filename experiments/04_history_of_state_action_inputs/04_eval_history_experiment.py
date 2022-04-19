r"""Script to evaluate experiment: influence of state-action history.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    05.10.2021
Updated:    27.10.2021 Adding eval functionalities for Run 02
"""
import os
import numpy as np
import shutil

# local imports
import phoenix_drone_simulation  # noqa
import phoenix_drone_simulation.utils.export as export


def main(args):
    current_file_dir = os.getcwd()
    data_path = os.path.join(current_file_dir, 'data/run_02')
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

    # === Create model.json files
    #   Sorted run_directories holds the following configurations:
    #       indices 0:15 - observation history = 1
    #       indices 15:30 - observation history = 2
    #       indices 30:45 - observation history = 4
    #       indices 45:60 - observation history = 6
    #       indices 60:75 - observation history = 8
    # for run_dir in list(sorted(run_directories))[60:75]:
    #     print(f'===\nCurrent Dir: {run_dir}')
    #     ac, env = utils.load_actor_critic_and_env_from_disk(run_dir)
    #     export.convert_actor_critic_to_json(actor_critic=ac,
    #                                         file_name_path=run_dir)

    # === Find the N = 10 best performing networks
    N = 10
    episode_return_scores = {}
    for run_dir in list(sorted(run_directories)):
        fnp = os.path.join(run_dir, 'returns.csv')
        ep_returns = export.get_file_contents(fnp)
        print(f'Mean ret: {np.mean(ep_returns):0.3f} from {fnp}')
        episode_return_scores[run_dir] = np.mean(ep_returns)

    best_N_runs = sorted(episode_return_scores.items(), key=lambda pair: pair[1], reverse=True)[:N]
    # max(stats.iteritems(), key=operator.itemgetter(1))[0]
    print('='*55, '\n')
    for i, (run_dir, ep_ret) in enumerate(best_N_runs):
        print(f'{i+1}. is {ep_ret} at {run_dir}')

    # === Copy best N `model.json` files to: /var/tmp/phoenix_nets
    print('='*55, '\nStart copying the N best networks...')
    target_dir = '/var/tmp/phoenix_nets'
    os.makedirs(target_dir, exist_ok=True)
    for i, (run_dir, ep_ret) in enumerate(best_N_runs):
        fnp_source = os.path.join(run_dir, 'model.json')
        new_model_json_name = f'exp_04_run_02_circle_task_model_{str(i+1).zfill(2)}.json'
        fnp_target = os.path.join(target_dir, new_model_json_name)
        print(f'Copy {fnp_source} ==> {fnp_target}')
        shutil.copyfile(fnp_source, fnp_target)


if __name__ == '__main__':
    # args, unparsed_args = get_training_command_line_args(
    #     alg=ALG, env=ENV, num_runs=NUM_RUNS)
    args = None
    main(args)
