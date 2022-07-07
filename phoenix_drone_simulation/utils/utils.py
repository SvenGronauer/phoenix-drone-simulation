r"""Utility functions for Project Phoenix-Simulation.

    Author:     Sven Gronauer
    Created:    17.05.2021
    Updated:    16.11.2021 Add algorithm helper functions
                16.04.2022 Added get_actor_critic_and_env_from_json_model
"""
from typing import Tuple

import numpy as np
import getpass
import json
import pandas
import torch
import torch.nn as nn
import scipy.sparse
import argparse
import datetime
import os
import sys
import gym
from collections import defaultdict
import re
from importlib import import_module

# local imports
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.algs import core


def convert_str_to_torch_functional(activation):
    r"""Converts a string to a non-linear activation function."""
    if isinstance(activation, str):  # convert string to torch functional
        activations = {
            'identity': nn.Identity,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'softplus': nn.Softplus,
            'tanh': nn.Tanh
        }
        assert activation in activations
        activation = activations[activation]
    assert issubclass(activation, torch.nn.Module)
    return activation


def extract_csr_matrix_from_data(data, layer_idx: int, csr_idx: int):
    csr_shape = data[str(layer_idx)][str(csr_idx)]["shape"]
    csr_data = data[str(layer_idx)][str(csr_idx)]["data"]
    csr_indices = data[str(layer_idx)][str(csr_idx)]["indices"]
    csr_indptr = data[str(layer_idx)][str(csr_idx)]["indptr"]
    return scipy.sparse.csr_matrix((csr_data, csr_indices, csr_indptr), shape=csr_shape)


def build_mlp_network(data, force_dense_matrices: False) -> torch.nn.Module:
    r"""Create multi-layer perceptron network with random init weights."""

    i = 0
    done = False
    layers = []
    activation = convert_str_to_torch_functional(data['activation'])
    while not done:
        # check if layer i exists in data dict
        if str(i) in data:

            if data[str(i)]['type'] == 'standard':
                # create a multi-layer perceptron layer
                weights = torch.squeeze(torch.Tensor([data[str(i)]['weights']]), dim=0)
                N, M = weights.shape
                # Note that M and N are changed!
                linear_layer = nn.Linear(M, N)
                # set values from data dict to pytorch linear layer module
                orig_size = linear_layer.weight.size()
                linear_layer.weight.data = weights.view(orig_size)
                bias = torch.squeeze(torch.Tensor([data[str(i)]['biases']]))
                linear_layer.bias.data = bias.view(-1)
            elif data[str(i)]['type'] == 'csrproduct':
                nb_csr_matrices = data[str(i)]['nb_csr_matrices']

                if force_dense_matrices:
                    res_mat = extract_csr_matrix_from_data(data=data, layer_idx=i, csr_idx=0)
                    for csr_idx in range(1, nb_csr_matrices):
                        res_mat = res_mat @ extract_csr_matrix_from_data(data=data, layer_idx=i, csr_idx=csr_idx)
                    res_mat = res_mat.todense()

                    weights = torch.nn.parameter.Parameter(torch.Tensor(res_mat))
                    M, N = weights.shape

                    linear_layer = nn.Linear(M, N)
                    linear_layer.weight = weights

                    bias = torch.squeeze(torch.Tensor([data[str(i)]['biases']]))
                    linear_layer.bias.data = bias.view(-1)
                else:
                    csr_matrices = [extract_csr_matrix_from_data(data=data, layer_idx=i, csr_idx=csr_idx) for csr_idx in range(nb_csr_matrices)]
                    bias = torch.squeeze(torch.Tensor([data[str(i)]['biases']])).view(-1)
                    linear_layer = SparseProductLayer(csr_matrices=csr_matrices, bias=bias)
            else:
                raise NotImplementedError('Only type=standard is supported.')

            # Add current layer to layers list
            if str(i+1) in data:  # use activation if not last layer
                layers += [linear_layer, activation()]
            else:  # last layer has no activation function (only identity)
                layers += [linear_layer, nn.Identity()]
                
            i += 1
        else:
            done = True

    assert layers is not [], 'Data dict does not hold layer information.'
    return nn.Sequential(*layers)


def get_actor_critic_and_env_from_json_model(
        json_fnp: str,
        env_id: str,
        algorithm: str = 'ppo' # look up default actor-critic values
) -> Tuple[torch.nn.Module, gym.Env]:
    r"""Loads a policy JSON file and creates an actor-critic Torch model.

    Note: Only the policy parameters are set, the critic still holds random
    parameter values.
    """
    actor_network = load_network_json(json_fnp)
    env = gym.make(env_id)
    actor_config = get_defaults_kwargs(alg=algorithm, env_id=env_id)

    ac = core.ActorCritic(
        actor_type=actor_config['actor'],  # Multi-layer perceptron Actor-critic Torch module
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_standardized_obs=True,
        use_scaled_rewards=True,
        use_shared_weights=False,
        ac_kwargs=actor_config['ac_kwargs']
    )

    # Set observation mean/standard-deviation parameter values
    data = get_file_contents(json_fnp)
    scaling_parameters = np.array(data['scaling_parameters'])
    ac.obs_oms.mean.data = torch.Tensor(scaling_parameters[0])
    ac.obs_oms.std.data = torch.Tensor(scaling_parameters[1])


    ac.pi.net = actor_network  # set and over-write
    ac.eval()  # set model to evaluation mode
    loggers.info('Created actor-critic model from JSON:')
    if loggers.MIN_LEVEL <= loggers.INFO:
        print(ac)
    return ac, env


def get_alg_module(alg, *submodules):
    """ inspired by source: OpenAI's baselines."""

    if submodules:
        mods = '.'.join(['phoenix_drone_simulation', 'algs', alg, *submodules])
        alg_module = import_module(mods)
    else:
        alg_module = import_module('.'.join(
            ['phoenix_drone_simulation', 'algs', alg, alg]))

    return alg_module


def get_alg_class(alg, env_id, **kwargs):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    alg_cls_init = getattr(alg_mod, 'get_alg')

    return alg_cls_init(env_id, **kwargs)


def get_default_args(debug_level=0,
                     env='CartPole-v0',
                     func_name='testing',
                     log_dir=f'/var/tmp/{getpass.getuser()}/',
                     threads=os.cpu_count()
                     ):
    """ create the default arguments for program execution
    :param threads: int, number of available threads
    :param env: str, name of RL environment
    :param func_name:
    :param log_dir: str, path to directory where logging files are going to be created
    :param debug_level:
    :return:
    """
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='This is the default parser.')
    parser.add_argument('--alg', default=os.cpu_count(), type=int,
                        help='Algorithm to use (in case of a RL problem. (default: PPO)')
    parser.add_argument('--threads', default=threads, type=int,
                        help='Number of available Threads on CPU.')
    parser.add_argument('--debug', default=debug_level, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--env', default=env, type=str,
                        help='Default environment for RL algorithms')
    parser.add_argument('--func', dest='func', default=func_name,
                        help='Specify function name to be testing')
    parser.add_argument('--log', dest='log_dir', default=log_dir,
                        help='Set the seed for random generator')

    args = parser.parse_args()
    args.log_dir = os.path.abspath(os.path.join(args.log_dir,
                                                datetime.datetime.now().strftime(
                                                    "%Y_%m_%d__%H_%M_%S")))
    return args


def get_seed_from_sys_args():
    _seed = 0
    for pos, elem in enumerate(
            sys.argv):  # look if there exists seed=X in _args,
        if len(elem) > 5 and elem[:5] == 'seed=':
            _seed = int(elem[5:])
    return _seed


def get_learn_function(alg):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    learn_func = getattr(alg_mod, 'learn')

    return learn_func


def get_env_type(env_id: str):
    """Determines the type of the environment if there is no args.env_type.

    source: OpenAI's Baselines Repository

    Parameters
    ----------
    env_id:
        Name of the gym environment.

    Returns
    -------
    env_type: str
    env_id: str
    """
    all_registered_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            all_registered_envs[env_type].add(env.id)
        except AttributeError:
            print('Could not fetch ', env.id)

    # Re-parse the gym registry, since we could have new data
    # since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        all_registered_envs[env_type].add(env.id)

    if env_id in all_registered_envs.keys():
        env_type = env_id
        env_id = [g for g in all_registered_envs[env_type]][0]
    else:
        env_type = None
        for g, e in all_registered_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id,
            all_registered_envs.keys())

    return env_type, env_id


def get_defaults_kwargs(alg, env_id):
    """ inspired by OpenAI's baselines."""
    env_type, _ = get_env_type(env_id=env_id)

    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        pass
        # warnings.warn(
        #     f'Could not fetch default kwargs for env_type: {env_type}')
        # Fetch standard arguments from locomotion environments
        try:  # fetch from defaults()
            env_type = 'defaults'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()
        except:
            env_type = 'locomotion'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()

    return kwargs

def get_policy_filename_path(
        file_name: str = "model_50_50_relu_PWM_circle_task.json"
) -> str:
    r"""Get the absolute path to a policy."""
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    # navigate to /examples/policies as directory....
    files_path = os.path.normpath(
        os.path.join(current_file_path, '../../data/policies'))
    fnp = os.path.join(files_path, file_name)
    loggers.debug(f'Get Policy Parameters from: {fnp}')
    return fnp

def load_network_json(
        file_name_path: str,
        force_dense_matrices=False
) -> torch.nn.Module:
    r"""Open the file with given path and return Python object."""
    assert os.path.isfile(file_name_path), \
        'No file exists at: {}'.format(file_name_path)
    assert file_name_path.endswith('.json'), 'Expected format is json.'

    data = get_file_contents(file_name_path)
    scaling_parameters = np.array(data['scaling_parameters'])

    net = build_mlp_network(data, force_dense_matrices=force_dense_matrices)

    # Calculate check-sum and compare with json file
    if hasattr(data, 'check_sum'):
        # Use a vector filled with ones to validate correct output of NN on hardware
        with torch.no_grad():
            out = net(torch.ones(scaling_parameters.shape[1]))
        assert np.allclose(np.sum(out.numpy()), data['check_sum']), \
            f"Checksum did not match. ({np.sum(out.numpy())} vs. " \
            f"{data['check_sum']})"
    else:
        loggers.warn(f'Could not validate `check_sum`: missing in JSON file.')


    loggers.info(loggers.colorize(f'Loaded JSON from: {file_name_path}', color='green', bold=True))

    return net


def get_file_contents(
        file_path: str,
        skip_header: bool = False
    ):
    r"""Open the file with given path and return Python object."""
    assert os.path.isfile(file_path), 'No file exists at: {}'.format(file_path)
    if file_path.endswith('.json'):  # return dict
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    elif file_path.endswith('.csv'):
        if skip_header:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        else:
            data = np.loadtxt(file_path, delimiter=",")
        if len(data.shape) == 2:  # use pandas for tables..
            data = pandas.read_csv(file_path)
    else:
        raise NotImplementedError
    return data


def dump_network_json(
        activation: str,
        scaling_parameters: np.ndarray,
        neural_network: torch.nn.Module,
        file_name_path: str,
        model_name="model",
        layer_indices_to_replace_with_csr_matrices=None,
        csr_matrices_for_replacement=None,
) -> None:
    """Dump neural network as JSON to disk.

    Uses the format defined by Matthias Kissel and Sven Gronauer.
    """

    assert (layer_indices_to_replace_with_csr_matrices is None) == (csr_matrices_for_replacement is None), "either layer_indices_to_replace_with_csr_matrices and csr_matrices_for_replacement must be both None or both not None"
    if not (layer_indices_to_replace_with_csr_matrices is None):
        assert len(layer_indices_to_replace_with_csr_matrices) == len(csr_matrices_for_replacement), "layer_indices_to_replace_with_csr_matrices and csr_matrices_for_replacement must have the same length"
    if layer_indices_to_replace_with_csr_matrices is None:
        layer_indices_to_replace_with_csr_matrices = []
        csr_matrices_for_replacement = []

    data = dict()
    # === Create Check Sum:
    # Use a vector filled with ones to validate correct output of NN on hardware
    with torch.no_grad():
        out = neural_network(torch.ones(scaling_parameters.shape[1]))
    # TODO: uncomment checksum line
    # data["check_sum"] = str(out.numpy().sum())
    data["scaling_parameters"] = scaling_parameters.tolist()
    # Write the activation function
    data["activation"] = activation

    data_entry_i = 0
    for layer_i, layer in enumerate(neural_network):
        if isinstance(layer, nn.Linear):
            if layer_i in layer_indices_to_replace_with_csr_matrices:
                replacement_idx = layer_indices_to_replace_with_csr_matrices.index(layer_i)
                data[str(data_entry_i)] = dict()
                data[str(data_entry_i)]["type"] = "csrproduct"
                data[str(data_entry_i)]["nb_csr_matrices"] = len(csr_matrices_for_replacement[replacement_idx])

                for csr_i, csr in enumerate(csr_matrices_for_replacement[replacement_idx]):
                    data[str(data_entry_i)][csr_i] = dict()
                    data[str(data_entry_i)][csr_i]["data"] = csr.data.tolist()
                    data[str(data_entry_i)][csr_i]["indices"] = csr.indices.tolist()
                    data[str(data_entry_i)][csr_i]["indptr"] = csr.indptr.tolist()
                    data[str(data_entry_i)][csr_i]["shape"] = [csr.shape[0], csr.shape[1]]

                biases = layer.bias.detach().cpu().numpy().tolist()
                data[str(data_entry_i)]["biases"] = biases
            else:
                data[str(data_entry_i)] = dict()
                data[str(data_entry_i)]["type"] = "standard"
                # get the weights
                weights = layer.weight.detach().cpu().numpy().tolist()
                data[str(data_entry_i)]["weights"] = weights
                # get the biases
                biases = layer.bias.detach().cpu().numpy().tolist()
                data[str(data_entry_i)]["biases"] = biases
            data_entry_i += 1
    # print(data)
    print('-' * 55)
    print(neural_network)
    # raise
    save_file_name_path = os.path.join(file_name_path, model_name + ".json")
    with open(save_file_name_path, 'w') as outfile:
        json.dump(data, outfile)
    print('-' * 55)
    print(f'JSON saved at: {save_file_name_path}')


def convert_actor_critic_to_json(
        actor_critic: torch.nn.Module,
        file_name_path
):
    """Save PyTorch Module as json to disk."""
    # Write the headers
    scaling_parameters = np.empty((2, 16))
    scaling_parameters[0] = actor_critic.obs_oms.mean.numpy()
    scaling_parameters[1] = actor_critic.obs_oms.std.numpy()
    # print(scaling_parameters)
    # raise NotImplementedError
    dump_network_json(
        activation=actor_critic.ac_kwargs['pi']['activation'],
        scaling_parameters=scaling_parameters,
        neural_network=actor_critic.pi.net,
        file_name_path=file_name_path
    )


def load_actor_critic_and_env_from_disk(
        file_name_path: str
) -> tuple:
    """Loads ac module from disk. (@Sven).

    Parameters
    ----------
    file_name_path

    Returns
    -------
    tuple
        holding (actor_critic, env)
    """
    config_file_path = os.path.join(file_name_path, 'config.json')
    conf = get_file_contents(config_file_path)
    print('Loaded config file:')
    print(conf)
    env_id = conf.get('env_id')
    env = gym.make(env_id)
    alg = conf.get('alg', 'ppo')

    if alg == 'sac':
        from phoenix_drone_simulation.algs.sac.sac import MLPActorCritic
        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            ac_kwargs=conf['ac_kwargs']
        )
    elif alg == "ddpg":
        from phoenix_drone_simulation.algs.ddpg.ddpg import MLPActorCritic
        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            ac_kwargs=conf['ac_kwargs']
        )
    else:
        ac = core.ActorCritic(
            actor_type=conf['actor'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=conf['use_standardized_obs'],
            use_scaled_rewards=conf['use_reward_scaling'],
            use_shared_weights=False,
            ac_kwargs=conf['ac_kwargs']
        )
    model_path = os.path.join(file_name_path, 'torch_save', 'model.pt')
    ac.load_state_dict(torch.load(model_path), strict=False)
    print(f'Successfully loaded model from: {model_path}')
    return ac, env


def test():
    file_name_path = "/Users/sven/Downloads/model_pwm_1.json"
    load_network_json(file_name_path=file_name_path)


class SparseProductLayer(nn.Module):
    def __init__(self, csr_matrices: list, bias: torch.tensor):
        super(SparseProductLayer, self).__init__()

        assert len(csr_matrices) > 0, "There must be at least one csr matrix in the list"
        self.sparse_matrices = nn.ParameterList([nn.Parameter(self.scipy_csr_matrix_to_torch_coo(csr_mat_scipy)) for csr_mat_scipy in csr_matrices])
        self.bias = bias

    def scipy_csr_matrix_to_torch_coo(self, csr_mat_scipy: scipy.sparse.csr_matrix):
        coo_mat_scipy = csr_mat_scipy.tocoo()
        
        values = coo_mat_scipy.data
        indices = np.vstack((coo_mat_scipy.row, coo_mat_scipy.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_mat_scipy.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, x):
        if len(x.shape) == 1:
            res = torch.unsqueeze(x, 0)
        else:
            res = x
            
        idx_list = list(range(len(self.sparse_matrices)))
        idx_list.reverse()
        for sparse_mat_idx in idx_list:
            res = torch.transpose(torch.sparse.mm(self.sparse_matrices[sparse_mat_idx], torch.transpose(res, 0, 1)), 0, 1)
        res = res + self.bias

        if len(x.shape) == 1:
            return torch.squeeze(res, dim=0)
        else:
            return res

if __name__ == '__main__':
    test()
