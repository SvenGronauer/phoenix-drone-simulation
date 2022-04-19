r"""Export functionalities for custum CrazyFlie Firmware.

"""
import os
import json
import gym
import numpy as np
import torch
import torch.nn as nn

# local imports
import phoenix_drone_simulation
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.utils.utils import get_file_contents


def count_vars(module: nn.Module):
    r"""Count number of variables in Neural Network."""
    return sum([np.prod(p.shape) for p in module.parameters()])


def dump_json(
        activation: str,
        scaling_parameters: np.ndarray,
        neural_network: torch.nn.Module,
        file_path: str,
        file_name: str = "model.json"
) -> None:
    """Dump neural network as JSON to disk.

    Uses the format defined by Matthias Kissel and Sven Gronauer.

    Parameters
    ----------
    activation
    scaling_parameters
    neural_network
    file_path
    file_name

    Returns
    -------
    None
    """
    data = dict()
    # === Create Check Sum:
    # Use a vector filled with ones to validate correct output of NN on hardware
    with torch.no_grad():
        out = neural_network(torch.ones(scaling_parameters.shape[1],
                                        dtype=torch.float32))
        print(f'Check sum: {out.numpy().sum()}')
    data["check_sum"] = str(out.numpy().sum())
    data["scaling_parameters"] = scaling_parameters.tolist()
    # Write the activation function
    data["activation"] = activation

    data_entry_i = 0
    for layer_i, layer in enumerate(neural_network):
        if isinstance(layer, nn.Linear):
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
    save_file_name_path = os.path.join(file_path, file_name)
    with open(save_file_name_path, 'w') as outfile:
        json.dump(data, outfile)
    print('-' * 55)
    # Count number of parameters in NN layers
    var_counts = tuple(count_vars(layer) for layer in neural_network)
    print(f'Number of parameters in Neural Net:\t{np.sum(var_counts)}\n')
    print(loggers.colorize(f'JSON saved at: {save_file_name_path}', color='green'))


def convert_actor_critic_to_json(actor_critic: torch.nn.Module,
                                 file_path: str,
                                 file_name: str = 'model.json'
                                 ):
    """Save PyTorch Module as json to disk."""
    input_dim = actor_critic.obs_oms.mean.numpy().shape[0]
    scaling_parameters = np.empty((2, input_dim))
    scaling_parameters[0] = actor_critic.obs_oms.mean.numpy()
    scaling_parameters[1] = actor_critic.obs_oms.std.numpy()
    dump_json(
        activation=actor_critic.ac_kwargs['pi']['activation'],
        scaling_parameters=scaling_parameters,
        neural_network=actor_critic.pi.net,
        file_path=file_path,
        file_name=file_name
    )


def convert_to_onxx_file_format(file_name_path):
    """ This method is used by Sven to convert his trained actor critic
    checkpoints to the ONXX format.

    """
    ac, env = load_actor_critic_and_env_from_disk(file_name_path)

    # Actor network is of shape:
    # --------------------------
    # Sequential(
    #   (0): Linear(in_features=12, out_features=64, bias=True)
    #   (1): Tanh()
    #   (2): Linear(in_features=64, out_features=64, bias=True)
    #   (3): Tanh()
    #   (4): Linear(in_features=64, out_features=4, bias=True)
    #   (5): Identity()
    # )
    print(ac.pi.net)
    #
    dummy_input = torch.ones(*env.observation_space.shape)
    # print('dummy_input: ', dummy_input)
    model = ac.pi.net

    print('=' * 55)
    print('Test ac.pi.net....')
    for name, param in model.named_parameters():
        print(name)
    print('=' * 55)

    torch.save(model.state_dict(), os.path.join(file_name_path, "ActorMLP.pt"))

    save_file_name_path = os.path.join(file_name_path, "ActorMLP.onnx")
    torch.onnx.export(model, dummy_input, save_file_name_path,
                      verbose=True,
                      # opset_version=12,  # ONNX version to export the model to
                      export_params=True,  # store the trained parameters
                      do_constant_folding=False,  # necessary to preserve parameters names!
                      # input_names=["input"],
                      # output_names=["output"]
    )

    # Save observation standardization
    print('=' * 55)
    print('Save observation standardization...')
    model = ac.obs_oms
    save_file_name_path = os.path.join(file_name_path, "ObsStand.onnx")
    torch.onnx.export(model, dummy_input, save_file_name_path,
                      verbose=True,
                      # opset_version=12,  # ONNX version to export the model to
                      export_params=True,  # store the trained parameters
                      do_constant_folding=False,  # necessary to preserve parameters names!
                      # input_names=["input"],
                      # output_names=["output"]
    )

