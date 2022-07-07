"""
    Define default parameters for SAC algorithm.
"""


def defaults():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (400, 300),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (400, 300),
                    'activation': 'relu'}
        },
        epochs=100,
        gamma=0.99,
    )


def locomotion():
    """Default hyper-parameters for Bullet's locomotion environments.

    Parameters are values suggested in:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    """
    params = defaults()
    params['mini_batch_size'] = 256
    params['buffer_size'] = 300000
    params['gamma'] = 0.98
    params['epochs'] = 500
    params['lr'] = 1e-3  # default choice is Adam
    return params


# Hack to circumvent kwarg errors with the official PyBullet Envs
def gym_locomotion_envs():
    params = locomotion()
    return params
