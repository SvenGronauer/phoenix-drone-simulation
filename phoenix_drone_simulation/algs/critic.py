""" Implementation of critic classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import torch
import torch.nn as nn

from phoenix_drone_simulation.algs.net import build_cascaded_network, \
    build_forward_network, build_recurrent_network

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
    def __init__(self, obs_dim, hidden_sizes, activation, shared=None):
        super(Critic, self).__init__()
        self.layers_rnn = []
        self.net = None

    def reset_states(self):
        for lay_rnn in self.layers_rnn:
            lay_rnn.reset_states()

    def forward(self, obs):
        assert self.net is not None
        return torch.squeeze(self.net(obs),
                             -1)  # Critical to ensure v has right shape.

    @property
    def is_recurrent(self):
        raise NotImplementedError


@register_critic("forward")
class ForwardCritic(Critic):

    def __init__(self, obs_dim, hidden_sizes, activation, layer='MLP'):
        super().__init__(obs_dim, hidden_sizes, activation)

        self.net, _ = build_forward_network(
            [obs_dim] + list(hidden_sizes) + [1],
            activation=activation)

    @property
    def is_recurrent(self):
        return False


@register_critic("recurrent")
class RecurrentCritic(Critic):

    def __init__(self, obs_dim, hidden_sizes, activation, layer='GRU'):
        super().__init__(obs_dim, hidden_sizes, activation)

        self.net, self.layers_rnn, _ = build_recurrent_network(
            [obs_dim] + list(hidden_sizes) + [1],
            activation=activation, layer=layer
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
