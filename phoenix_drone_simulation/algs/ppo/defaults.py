"""
    Define default parameters for Proximal Policy Optimization (PPO) algorithm.
"""



def defaults():
    return dict(
        actor='recurrent',
        ac_kwargs={
            'pi': {
                'activation': 'identity',
                'hidden_sizes': [18, 18],
                'layer': 'GRU'
            },
            'val': {
                'activation': 'identity',
                'hidden_sizes': [128, 128],
                'layer': 'GRU'
            }
        },
        adv_estimation_method='gae',
        critic='recurrent',
        epochs=300,  # 3.2M steps
        gamma=0.99,
        steps_per_epoch=32 * 1000
    )


def locomotion():
    """Default hyper-parameters for Bullet's locomotion environments."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 1000
    params['pi_lr'] = 3e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000
    params['vf_lr'] = 3e-4  # default choice is Adam
    return params


# Hack to circumvent kwarg errors with the official PyBullet Envs
def gym_locomotion_envs():
    params = locomotion()
    return params


def gym_manipulator_envs():
    """Default hyper-parameters for Bullet's manipulation environments."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 150
    params['pi_lr'] = 3e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000
    params['vf_lr'] = 3e-4  # default choice is Adam
    return params
