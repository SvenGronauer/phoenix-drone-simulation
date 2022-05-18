"""
    Define default parameters for NPG algorithm.
"""
import phoenix_drone_simulation.utils.mpi_tools as mpi


def defaults():
    return dict(
        actor='mlp',
        ac_kwargs={
            'pi': {'hidden_sizes': (400, 300),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (400, 300),
                    'activation': 'relu'}
        },
        epochs=100,
        gamma=0.99,
        mini_batch_size=128
    )


def locomotion():
    """Default hyper-parameters for Bullet's locomotion environments.

    These parameters are in line with the suggestions in RL Baselines Zoo.:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ddpg.yml
    """
    params = defaults()
    params['act_noise'] = 0.1
    params['buffer_size'] = 200000
    params['gamma'] = 0.98
    params['epochs'] = 500
    params['update_after'] = 10000
    params['update_every'] = 64
    params['pi_lr'] = 1e-3  # default choice is Adam
    params['q_lr'] = 1e-3  # default choice is Adam
    return params


# Hack to circumvent kwarg errors with the official PyBullet Envs
def gym_locomotion_envs():
    params = locomotion()
    return params
