"""Train drone hover task with Stable-Baselines (SB) 3 framework

Author:     Sven Gronauer
Created:    12.05.2021
"""
import os
import gym
import phoenix_drone_simulation  # necessary to load our custom drone environments
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # === 1) Train model in source domain (System Equations)
    env_source = gym.make("DroneHoverSimpleEnv-v0")
    model_source_domain = PPO("MlpPolicy", env_source, verbose=1)
    # at least 1.000.000 total time steps are recommended
    model_source_domain.learn(total_timesteps=int(1e4))

    # Save the agent to disk
    os.makedirs('/var/tmp', exist_ok=True)
    model_source_domain.save("/var/tmp/model_source")

    # Evaluate trained policy in source domain
    mean_reward, std_reward = evaluate_policy(
        model_source_domain, env_source, n_eval_episodes=10, deterministic=True)
    print(f"Before Transfer:\nmean_reward = {mean_reward:.2f} +/- {std_reward}")

    # === 2) Transfer to target domain (Bullet Simulation)
    print('='*50, '\nTARGET DOMAIN')
    env_target = gym.make("DroneHoverBulletEnv-v0")

    # apply once domain randomization and then deactivate it
    env_target.apply_domain_randomization()
    env_target.unwrapped.domain_randomization = -1

    # Evaluate pre-trained policy in target domain
    mean_reward, std_reward = evaluate_policy(
        model_source_domain, env_target, n_eval_episodes=50, deterministic=True)
    print(f"After Transfer:\nmean_reward = {mean_reward:.2f} +/- {std_reward}")

    # now re-use the model from the source domain
    model_target_domain = PPO("MlpPolicy", env_target, verbose=1)
    # load parameters from the source domain model
    model_target_domain.load("/var/tmp/model_source")

    # Start training
    model_target_domain.learn(total_timesteps=int(1e4))
    mean_reward, std_reward = evaluate_policy(
        model_target_domain, env_target, n_eval_episodes=50, deterministic=True)
    print(f"After Post-training:\nmean_reward = {mean_reward:.2f} +/- {std_reward}")


if __name__ == '__main__':
    main()
