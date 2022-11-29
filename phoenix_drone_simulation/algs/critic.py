""" Implementation of critic classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
from typing import List

import torch
import torch.nn as nn

from phoenix_drone_simulation.algs.net import build_cascaded_network, \
    build_forward_network, StatefulRNN, build_network

registered_critics = dict()  # global dict that holds pointers to functions


def register_critic(critic_name):
    """ register critic into global dict """

    def wrapper(func):
        registered_critics[critic_name] = func
        return func

    return wrapper


def get_registered_critic_fn(critic_type: str):
    critic_fn = critic_type
    msg = f'Did not find: {critic_fn} in registered critics.'
    assert critic_fn in registered_critics, msg
    return registered_critics[critic_fn]


# ====================================
#       Critic Modules
# ====================================

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = None

    def forward(self, obs):
        assert self.net is not None
        return torch.squeeze(self.net(obs),
                             -1)  # Critical to ensure v has right shape.

    @property
    def is_recurrent(self):
        raise NotImplementedError

    @property
    def recurrent_layers(self) -> List[StatefulRNN]:
        rnn_layers = []
        for layer in self.net:
            if isinstance(layer, StatefulRNN):
                rnn_layers.append(layer)
        return rnn_layers

    def reset_states(self):
        for lay_rnn in self.recurrent_layers:
            lay_rnn.reset_states()


# @register_critic("forward")
# class ForwardCritic(Critic):
#
#     def __init__(self, obs_dim, hidden_sizes, activation, layer='MLP'):
#         super().__init__()
#
#         self.net, _ = build_forward_network(
#             [obs_dim] + list(hidden_sizes) + [1],
#             activation=activation)
#
#     @property
#     def is_recurrent(self):
        return False


@register_critic("recurrent")
class RecurrentCritic(Critic):

    def __init__(self, obs_dim, hidden_sizes, layers):
        super().__init__()

        # self.net, self.layers_rnn, _ = build_recurrent_network(
        #     [obs_dim] + list(hidden_sizes) + [1],
        #     activation=activation, layer=layer
        # )

        self.net = build_network(
            sizes=[obs_dim] + list(hidden_sizes) + [1],
            layers=layers,
        )

    @property
    def is_recurrent(self):
        return True


@register_critic("cascaded")
class CascadedCritic(Critic):

    def __init__(self, obs_dim, hidden_sizes, activation, layer='GRU'):
        super().__init__(obs_dim, hidden_sizes, activation)

        self.net, self.layers_rnn, _ = build_cascaded_network(
            [obs_dim] + list(hidden_sizes) + [1],
            activation=activation
        )

    @property
    def is_recurrent(self):
        return True
