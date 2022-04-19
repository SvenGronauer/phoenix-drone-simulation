"""Loads a pre-trained NN model which lets the drone fly in a circle.

Author:     Sven Gronauer
Created:    16.04.2022
"""
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.play import play_after_training


def main():
    env_id = 'DroneCircleBulletEnv-v0'
    policy_name = "model_50_50_relu_PWM_circle_task.json"
    json_fnp = utils.get_policy_filename_path(policy_name)
    ac, env = utils.get_actor_critic_and_env_from_json_model(json_fnp, env_id)

    play_after_training(
        actor_critic=ac,
        env=env,
        noise=False
    )


if __name__ == '__main__':
    main()

