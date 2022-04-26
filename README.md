# Phoenix-Drone-Simulation 


An OpenAI [Gym environment](https://gym.openai.com/envs/#classic_control) based 
on [PyBullet](https://github.com/bulletphysics/bullet3) for learning to control 
the CrazyFlie quadrotor: 

- Can be used for Reinforcement Learning (check out the examples!) or Model 
  Predictive Control
- We used this repository for sim-to-real transfer experiments (see publication [1] below)
- The implemented dynamics model is based on the [Bitcraze's Crazyflie 2.1 nano-quadrotor](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf)


Circle Task | TakeOff
--- |  ---
![Circle](./docs/readme/circle3.gif) |![TakeOff](./docs/readme/takeoff.gif)


The following tasks are currently available to fly the little drone:
- Hover
- Circle 
- Take-off *(implemented but not yet working properly: reward function must be tuned!)*
- ~~Reach~~ (not yet implemented)


## Overview of Environments

|                                       | Task         | Controller    | Physics            | Observation Frequency | Domain Randomization |  *Aerodynamic effects*     |  Motor Dynamics   |
|-------------------------------------: | :----------: | :-----------: | :----------------: | :-------------------: | :------------------: | :------------------------: | :---------------: | 
| `DroneHoverSimpleEnv-v0`              | Hover        | PWM (100Hz)   | Simple             | 100 Hz                | 10%                  | None                        | Instant force     |
| `DroneHoverBulletEnv-v0`              | Hover        | PWM (100Hz)   | PyBullet           | 100 Hz                | 10%                  | None                        | First-order       |
| `DroneCircleSimpleEnv-v0`             | Circle       | PWM (100Hz)   | Simple             | 100 Hz                | 10%                  | None                        | Instant force     |
| `DroneCircleBulletEnv-v0`             | Circle       | PWM (100Hz)   | PyBullet           | 100 Hz                | 10%                  | None                        | First-order     |
| `DroneTakeOffSimpleEnv-v0`             | Take-off     | PWM (100Hz)   | Simple             | 100 Hz                | 10%                  | Ground-effect              | Instant force     |
| `DroneTakeOffBulletEnv-v0`             | Take-off     | PWM (100Hz)   | PyBullet           | 100 Hz                | 10%                  | Ground-effect              | First-order     |

# Installation and Requirements

Here are the (few) steps to follow to get our repository ready to run. Clone the
repository and install the phoenix-drone-simulation package via pip. Note that 
everything after a `$` is entered on a terminal, while everything after `>>>` 
is passed to a Python interpreter. Please, use the following three steps for 
installation:
```
$ git clone https://github.com/SvenGronauer/phoenix-drone-simulation
$ cd phoenix-drone-simulation/
$ pip install -e .
```

This package follows OpenAI's [Gym Interface](https://github.com/openai/gym/blob/master/docs/creating-environments.md).

> Note: if your default `python` is 2.7, in the following, replace `pip` with `pip3` and `python` with `python3`


## Supported Systems

We tested this package under *Ubuntu 20.04* and *Mac OS X 11.2* running Python 
3.7 and 3.8. Other system might work as well but have not been tested yet.
Note that PyBullet supports Windows as platform only experimentally!. 


## Dependencies 

Bullet-Safety-Gym heavily depends on two packages:

+ [Gym](https://github.com/openai/gym)
+ [PyBullet](https://github.com/bulletphysics/bullet3)


## Getting Started


After the successful installation of the repository, the Bullet-Safety-Gym 
environments can be simply instantiated via `gym.make`. See: 

```
>>> import gym
>>> import phoenix_drone_simulation
>>> env = gym.make('DroneHoverBulletEnv-v0')
```

The functional interface follows the API of the OpenAI Gym (Brockman et al., 
2016) that consists of the three following important functions:

```
>>> observation = env.reset()
>>> random_action = env.action_space.sample()  # usually the action is determined by a policy
>>> next_observation, reward, done, info = env.step(random_action)
```

A minimal code for visualizing a uniformly random policy in a GUI, can be seen 
in:

```
import gym
import time
import phoenix_drone_simulation

env = gym.make('DroneHoverBulletEnv-v0')

while True:
    done = False
    env.render()  # make GUI of PyBullet appear
    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
        time.sleep(0.05)
```
Note that only calling the render function before the reset function triggers 
visuals.

# Training Policies

To train an agent with the PPO algorithm call:
```
$ python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnv-v0
```

This works with basically every environment that is compatible with the OpenAI 
Gym interface:
```
$ python -m phoenix_drone_simulation.train --alg ppo --env CartPole-v0
```

After an RL model has been trained and its checkpoint has been saved on your 
disk, you can visualize the checkpoint:
```
$ python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT
```
where PATH_TO_CKPT is the path to the checkpoint, e.g.
`/var/tmp/sven/DroneHoverSimpleEnv-v0/trpo/2021-11-16__16-08-09/seed_51544`

# Examples

### `generate_trajectories.py`

See the `generate_trajectories.py` script which shows how to generate data 
batches of size N. Use `generate_trajectories.py --play` to visualize the policy
in PyBullet simulator. 

### `train_drone_hover.py`

Use Reinforcement Learning (RL) to learn the drone holding its position at (0, 0, 1). 
This canonical example relies on the [RL-safety-Algorithms](https://github.com/SvenGronauer/RL-Safety-Algorithms) 
repository which is a very strong framework for parallel RL algorithm training.
 
### `transfer_learning_drone_hover.py`

Shows a transfer learning approach. We first train a PPO model in the source domain 
`DroneHoverSimpleEnv-v0` and then re-train the model on a more complex target 
domain `DroneHoverBulletEnv-v0`.
Note that the `DroneHoverBulletEnv-v0` environment builds upon an accurate 
motor modelling of the CrazyFlie drone and includes a motor dead time as well as
a motor lag.


# Tools

- `convert.py` @ Sven Gronauer

A function used by Sven to extract the policy networks from
his trained Actor Critic module and convert the model to a json file format.



# Version History and Changes


| Version | Changes | Date |
|-------: | :----------------: |  :----------------: |
| v1.0     | *Public Release*: Simulation parameters as proposed in Publication [1]  | 19.04.2022 | 
| v0.2     | Add: accurate motor dynamic model and first real-world transfer insights | 21.09.2021 | 
| v0.1     | Re-factor: of repository  (only Hover task yet implemented)  | 18.05.2021 | 
| v0.0     | Fork: from [Gym-PyBullet-Drones Repo](https://github.com/utiasDSL/gym-pybullet-drones)  | 01.12.2020 | 


# Publications

1.  Using Simulation Optimization to Improve Zero-shot Policy Transfer of Quadrotors
    
    *Sven Gronauer, Matthias Kissel, Luca Sacchetto, Mathias Korte, Klaus Diepold*
    
    https://arxiv.org/abs/2201.01369



-----
Lastly, we want to thank:
- Jacopo Panerati and his team for contributing the [Gym-PyBullet-Drones Repo](https://github.com/utiasDSL/gym-pybullet-drones) 
  which was the staring point for this repository.

- Artem Molchanov and collaborators for their hints about the CrazyFlie Firmware and the motor dynamics in their paper "Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors"

- Jakob Foerster for this Bachelor Thesis and his insights about the CrazyFlie's parameter values



-----
This repository has been develepod at the
> [Chair of Data Processing](https://www.ce.cit.tum.de/en/ldv/homepage/)             
> TUM School of Computation, Information and Technology                    
> Technical University of Munich        
