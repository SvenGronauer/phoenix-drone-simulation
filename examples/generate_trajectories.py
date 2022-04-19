import argparse
from phoenix_drone_simulation.utils.trajectory_generator import get_generator
import numpy as np


def generate_trajectories(env_name):

    generator = get_generator(env_name)

    performance = generator.evaluate(num_trajectories=10)
    print(f'J: {performance}')

    X, Y = generator.get_batch(1000)
    print(X.shape)
    print(Y.shape)

    print(np.std(X, axis=0))


def render_policy_trajectories(env_name):
    generator = get_generator(env_name)
    generator.play_policy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--env', type=str, default='DroneCircleBulletEnv-v0',
                        help='Name of the environment to be evaluated.')
    parser.add_argument('--play', action='store_true',
                        help='Visualize agent based on policy from ONNX.')
    args = parser.parse_args()

    if args.play:
        # Visualize the policy from an ONNX file in PyBullet's GUI
        render_policy_trajectories(env_name=args.env)

    else:
        # Generates input-output pairs depending on environment and policy
        generate_trajectories(env_name=args.env)

