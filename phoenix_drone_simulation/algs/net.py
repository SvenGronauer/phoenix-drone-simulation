""" Implementation of custom network classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn

mlp_activations = {
    'identity': nn.Identity,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'softplus': nn.Softplus,
    'tanh': nn.Tanh
}

recurrent_layers = {
    'gru': nn.GRU,
    'lstm': nn.LSTM
}


def initialize_layer(
        init_function: str,
        layer: torch.nn.Module
):
    if init_function == 'kaiming_uniform':  # this the default!
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    # glorot is also known as xavier uniform
    elif init_function == 'glorot' or init_function == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':  # matches values from baselines repo.
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise NotImplementedError


def convert_str_to_torch_functional(activation):
    if isinstance(activation, str):  # convert string to torch functional
        assert activation in mlp_activations
        activation = mlp_activations[activation]
    assert issubclass(activation, torch.nn.Module)
    return activation


def convert_str_to_torch_layer(layer):
    if isinstance(layer, str):  # convert string to torch layer
        assert layer in recurrent_layers
        layer = recurrent_layers[layer]
    assert issubclass(layer, torch.nn.Module)
    return layer


class StatefulRNN(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.state = None

    def reset_states(self):
        self.state = None

    def forward(self, x):
        inp_size = x.size()
        if len(inp_size) < 3:
            # Bring data into Format: [Batch,Seq,Feat]
            x = x.view(1, -1, x.size()[-1])
        y, state = self.layer(x, self.state)
        self.state = self._detach_state(state)
        y_size = list(inp_size)
        y_size[-1] = y.size()[-1]
        y = y.view(y_size)
        return y

    def _detach_state(self, state):
        # Detach hidden state from gradient computation and replace nan's with 0
        if isinstance(state, tuple):
            return tuple(torch.nan_to_num(s.detach()) for s in state)
        if isinstance(state, list):
            return [torch.nan_to_num(s.detach()) for s in state]
        return torch.nan_to_num(state.detach())


# @DeprecationWarning
# def build_recurrent_network(
#         sizes,
#         activation='identity',
#         output_activation='identity',
#         weight_initialization='kaiming_uniform',
#         layer='GRU'
# ):
#     tf_model = None
#     layer = convert_str_to_torch_layer(layer)
#     activation = convert_str_to_torch_functional(activation)
#     output_activation = convert_str_to_torch_functional(output_activation)
#     layers = list()
#     layers_rnn = []
#     for j in range(len(sizes) - 1):
#         act = activation if j < len(sizes) - 2 else output_activation
#         lay_stateless = layer(sizes[j], sizes[j + 1],
#                               batch_first=True) if j < len(sizes) - 2 else None
#         if lay_stateless is None:
#             lay_affine = nn.Linear(sizes[j], sizes[j + 1])
#             initialize_layer(weight_initialization, lay_affine)
#             lay_statefull = lay_affine
#         else:
#             lay_statefull = StatefulRNN(lay_stateless)
#             layers_rnn.append(lay_statefull)
#         layers += [lay_statefull, act()]
#     built_net = nn.Sequential(*layers)
#     print(built_net)
#     return built_net, layers_rnn, tf_model


def build_network(
        sizes: List[int],  # in_dim, h_dim, h_dim, out_dim, e.g. [10, 50, 50, 4]
        layers: List[str],  # e.g. ["tanh, tanh", "tanh"]
        weight_initialization='kaiming_uniform',
) -> nn.Module:
    if (len(sizes)-2) == len(layers):  # identitiy as last output activation
        layers.append("identity")
    assert (len(sizes)-1) == len(layers)
    layer_list = list()
    for j in range(len(sizes) - 1):

        if layers[j].lower() in mlp_activations:  # this is an affine layer
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            initialize_layer(weight_initialization, affine_layer)
            act = convert_str_to_torch_functional(mlp_activations[layers[j].lower()])
            layer_list += [affine_layer, act()]
        elif layers[j].lower() in recurrent_layers:
            rnn_fn = recurrent_layers[layers[j].lower()]
            lay_stateless = rnn_fn(sizes[j], sizes[j + 1], batch_first=True)
            recurrent_layer = StatefulRNN(lay_stateless)
            layer_list += [recurrent_layer, nn.Identity()]
        else:
            raise ValueError(f"Did not find: {layers[j]} as layer!")

    built_net = nn.Sequential(*layer_list)
    # print(built_net)
    return built_net


class CascadedNN(nn.Module):
    def __init__(self, layer_out, layer_in1, layer_in2=None):
        super().__init__()
        self.layer_in1 = layer_in1
        self.layer_in2 = layer_in2
        self.layer_out = layer_out

    def forward(self, x):
        y_1 = self.layer_in1(x)
        if self.layer_in2 is None:
            y_2 = x
        else:
            y_2 = self.layer_in2(x)
        x_12 = torch.cat((y_1, y_2), -1)
        y = self.layer_out(x_12)
        return y


def build_cascaded_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU'
):
    tf_model = None
    layer = convert_str_to_torch_layer(layer)
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    net = None
    layers_rnn = []
    for j in range((len(sizes) - 2) // 2):
        lay_rnn = StatefulRNN(
            layer(sizes[j * 2], sizes[j * 2 + 1], batch_first=True))
        layers_rnn.append(lay_rnn)
        lay_lin = nn.Linear(sizes[j * 2 + 1] + sizes[j * 2], sizes[(j + 1) * 2])
        initialize_layer(weight_initialization, lay_lin)
        net_pre = net
        net = nn.Sequential(CascadedNN(lay_lin, lay_rnn, net_pre), activation())
    return (
        nn.Sequential(net, nn.Linear(sizes[-2], sizes[-1]),
                      output_activation()),
        layers_rnn)


def build_forward_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform'
):
    tf_model = None
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)
