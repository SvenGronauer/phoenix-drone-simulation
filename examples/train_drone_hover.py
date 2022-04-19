"""Train drone hover task.

Note: this Python script utilizes only one CPU core. For multi-core training
please have a look at: train_with_multi_cores.py

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    12.05.2021
Updated:    16.11.2021 use algorithms from phoenix_drone_simulation.algs
"""
import gym
import getpass
import time
import torch

# local imports:
import phoenix_drone_simulation  # necessary to load our custom drone environments
from phoenix_drone_simulation.algs.model import Model


def main():
    env_id = "DroneHoverBulletEnv-v0"
    user_name = getpass.getuser()

    # Create a seed for the random number generator
    random_seed = int(time.time()) % 2 ** 16

    # I usually save my results into the following directory:
    default_log_dir = f"/var/tmp/{user_name}"

    # NEW: use algorithms implemented in phoenix_drone_simulation:
    # 1) Setup learning model
    model = Model(
        alg='ppo',  # choose between: trpo, ppo
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )
    model.compile()

    # 2) Train model - it takes typically at least 100 epochs for training
    model.fit(epochs=25)

    # 3) Benchmark the final policy and save results into `returns.csv`
    model.eval()

    # 4) visualize trained PPO model
    env = gym.make(env_id)
    env.render()
    # Important note:   PyBullet necessitates to call env.render() before
    #  env.reset() to display the GUI!
    while True:
        obs = env.reset()
        done = False
        while not done:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            action, value, *_ = model.actor_critic(obs)
            obs, reward, done, info = env.step(action)

            time.sleep(1/60)
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()

