r"""Objective function for simulation optimization with Drone Hover task.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    20-08-2021
Updated:    22-11-2021 Fixed observations such that sim is equal to real
            28-10-2021 Added ObjectiveFunction for Circle and Hover task
"""
import abc
import pybullet as pb
import os
import gym
import numpy as np
from typing import Optional

# local imports
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.simopt.core import ObjectiveFunctionBase, \
    RealWorldDataBuffer, DataBufferBase
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.circle import DroneCircleBulletEnv
import phoenix_drone_simulation.utils.mpi_tools as mpi
import phoenix_drone_simulation.simopt.config as so_conf
from phoenix_drone_simulation.simopt.plot_utils import plot_trajectories


class ObjectiveFunctionPyBullet(ObjectiveFunctionBase, abc.ABC):
    def __init__(
            self,
            files_path: str,
            seed: Optional[int] = None
    ):
        super().__init__(seed=seed, files_path=files_path)
        default_parameters = np.array([
            self.sim_env.drone.thrust_to_weight_ratio,  # [no unit]
            self.sim_env.drone.MOTOR_TIME_CONSTANT,  # in [s]
            self.sim_env.drone.LATENCY,  # in [s]
            # self.sim_env.drone.J[0, 0],  # J_x
            # self.sim_env.drone.J[1, 1],  # J_y
            # self.sim_env.drone.J[2, 2],  # J_z
        ])
        self.params = default_parameters.copy()
        low_boundary = np.array([
            1.5,  # [no unit]
            0.010,  # in [s]
            0.000,  # in [s]
        ])
        high_boundary = np.array([
            2.5,  # [no unit]
            0.500,  # in [s]
            0.050,  # in [s]
        ])
        # define param space to check the range of newly suggested parameters
        self.parameter_space = gym.spaces.Box(
            low=low_boundary,
            high=high_boundary
        )

    def _load_real_world_data(self) -> DataBufferBase:
        r"""Loads logged real-world data from hardware disk."""
        loggers.info(f'Load data from: {self.files_path}')
        return RealWorldDataBuffer(self.files_path)

    def check_parameters(self, params) -> bool:
        r"""Check range and dimensionality of parameters."""
        assert self.parameter_space.contains(params), \
            f'Parameters out of bounds:\n' \
            f'Got \t \t: \t{params}\n' \
            f'Lower bounds: \t{self.parameter_space.low}\n' \
            f'Upper bounds: \t{self.parameter_space.high}\n'
        return True

    def evaluate(self,
                 params: np.ndarray,
                 shrink: int = 1,
                 shuffle: bool = True,  # shuffle data set
                 ) -> float:
        r"""Evaluate the performance (fitness) of the suggested parameters.

        Parameters
        ----------
        params:
            Parameter vector
        shrink:
            Smaller data set size by this factor, e.g. used for Nesterov's
            Gaussian finite-difference method
        shuffle:
             Shuffle data set for mini-batch calculations

        Returns
        -------
        Score of objective function evaluation
        """
        self.set_parameters(params)

        # observation data set is of shape: (N, T, 12)
        pid = mpi.proc_id()
        N = self.real_data.observations.shape[0]
        K = N // shrink  # decrease data batch
        M = int(K // mpi.num_procs())
        indices = np.arange(N)
        assert M > 0

        # Shuffle data set for mini-batch calculations
        # note: makes only sense when taking less data as original data set
        if shuffle and shrink > 1:
            np.random.shuffle(indices)
            mpi.broadcast(indices)  # ensure identical index order for all procs

        # step over data batch: each process gets 1/N share
        rets = list()
        start = pid * M
        end = start + M
        _slice = slice(start, end)

        # now step over each sample of the process batch
        for idx_sample in indices[_slice]:
            obs = self.real_data.observations[idx_sample]
            acs = self.real_data.actions[idx_sample]
            pre_inputs = self.real_data.pre_inputs[idx_sample]
            ret = self.evaluate_once(obs, acs, pre_inputs=pre_inputs)
            rets.append(ret)

        # average returns over all processes
        averaged_performance = mpi.mpi_avg(np.mean(rets))
        return float(averaged_performance)

    def evaluate_once(self, obs, acs, pre_inputs) -> float:
        r"""Evaluates one mini-trajectory of size T."""
        errs = []
        gamma = 0.95  # weighting factor: discount future trajectory states
        # 1) do pre-steps for motor dynamics..
        self.sim_env.reset()
        for j in range(pre_inputs.shape[0]):
            u = pre_inputs[j]
            self.sim_env.step(u)
        x = self.sim_env.drone.x.copy()
        y = self.sim_env.drone.y.copy()

        # 2) set initial state of drone according to real-world observation
        # Layout in CSV files:
        # x,y,z,x_dot,y_dot,z_dot,roll,pitch,yaw,roll_dot,pitch_dot,yaw_dot
        x0 = obs[0]
        self.sim_env.init_xyz = x0[:3]
        self.sim_env.init_rpy = rpy = x0[6:9]
        self.sim_env.init_quaternion = q = np.array(pb.getQuaternionFromEuler(rpy))
        self.sim_env.init_xyz_dot = x0[3:6]

        # PyBullet expects RPY_dot in world coordinates
        R = np.array(pb.getMatrixFromQuaternion(q)).reshape((3, 3))
        self.sim_env.init_rpy_dot = R @ x0[9:12]

        # 3) reset simulation and use initial parameters from prior lines
        self.sim_env.enable_reset_distribution = False
        self.sim_env.reset()
        self.sim_env.drone.x = x
        self.sim_env.drone.y = y

        xs_sim = [self.sim_env.observation_history[-1].copy(), ]

        # 4) check if observations are equal
        # (here, linear and angular velocities are compared)
        # assert np.testing.assert_allclose(pb.getEulerFromQuaternion(x_sim[3:7]),
        #                                   x0[3:6], rtol=5e-2)
        # assert np.testing.assert_allclose(x_sim[7:13], x0[6:12])

        # 5) Determine error
        T = self.real_data.mini_trajectory_size
        for i in range(T-1):
            u = acs[i]
            self.sim_env.step(u)
            # get latest observation from history:
            sim_obs = self.sim_env.observation_history[-1].copy()
            x_real_next = obs[i+1]

            L = self.loss_function(obs_sim=sim_obs, obs_real=x_real_next)

            errs.append(gamma ** i * L)
            xs_sim.append(sim_obs)

        # Uncomment next line to display mini-trajectories
        # plot_trajectories(obs_real=obs, obs_sim=np.array(xs_sim), actions=acs)

        return float(np.mean(errs))

    def get_parameters(self) -> np.ndarray:
        return np.array([
            self.sim_env.drone.thrust_to_weight_ratio,
            self.sim_env.drone.T,
            self.sim_env.drone.latency,
            # self.sim_env.drone.J[0, 0],
            # self.sim_env.drone.J[1, 1],
            # self.sim_env.drone.J[2, 2],
        ])

    @classmethod
    def loss_function(cls,
                      obs_sim: np.ndarray,
                      obs_real: np.ndarray
                      ) -> float:
        r"""Computes the distance between observations from sim and real."""

        # angles
        a = so_conf.quaternion_sim_slice
        b = so_conf.rpy_real_slice  # in [rad]
        e_rpy = np.array(pb.getEulerFromQuaternion(obs_sim[a])) - obs_real[b]

        # angle rates
        a = so_conf.rpy_dot_sim_slice  # in [rad/s]
        b = so_conf.rpy_dot_real_slice  # in [rad/s]
        err_rpy_dot = obs_sim[a] - obs_real[b]

        # position - errors are smaller than angle errors
        a = so_conf.xyz_sim_slice
        b = so_conf.xyz_real_slice
        e_xyz = 100 * (obs_sim[a] - obs_real[b])

        # linear velocity
        a = so_conf.xyz_dot_sim_slice
        b = so_conf.xyz_dot_real_slice
        e_xyz_dot = 10 * (obs_sim[a] - obs_real[b])

        # Build norms of error vector:
        err = np.hstack((e_rpy, e_xyz, e_xyz_dot, err_rpy_dot))
        L1 = np.linalg.norm(err, ord=1)
        L2 = np.linalg.norm(err, ord=2)
        L = L1 + L2
        return L

    def sample(self) -> np.ndarray:
        r"""Samples a random parameter vector from a valid parameter range."""
        return self.parameter_space.sample()

    def set_parameters(self, params):
        """Set parameters to simulation env.

        Note: PyBullet must be stepped after calling this method to make
            parameter changes permanent to simulation.
        """
        # assert self.check_parameters(params)
        positive_params = np.clip(params, 0, np.inf)
        self.params = positive_params

        # set A, B, K according to new values of T_s, T
        self.sim_env.drone.update_motor_dynamics(
            new_motor_time_constant=self.params[1],
            new_thrust_to_weight_ratio=self.params[0]
        )
        self.sim_env.drone.set_latency(self.params[2])


class ObjectiveFunctionHoverTask(ObjectiveFunctionPyBullet):
    def __init__(
            self,
            seed: Optional[int] = None
    ):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        files_path = os.path.join(current_file_path,
                                  '../../data/sim_opt/hover_task')
        super().__init__(seed=seed, files_path=files_path)

    def _load_simulation(self) -> gym.Env:
        r"""Creates an instance of the simulation environment.."""
        _env = DroneHoverBulletEnv()
        # Note: If you may wish to disable Domain Randomization (DR):
        _env.domain_randomization = -1  # disable DR
        _env.observation_noise = -1  # disable DR
        # seeding
        seed = self.seed
        seed += 10000 * mpi.proc_id()
        np.random.seed(seed)
        _env.seed(seed)
        return _env


class ObjectiveFunctionCircleTask(ObjectiveFunctionPyBullet):
    def __init__(
            self,
            seed: Optional[int] = None
    ):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        files_path = os.path.join(current_file_path,
                                  '../../data/sim_opt/circle_task')
        super().__init__(seed=seed, files_path=files_path)

    def _load_simulation(self) -> gym.Env:
        r"""Creates an instance of the simulation environment.."""
        _env = DroneCircleBulletEnv()
        # Note: If you may wish to disable Domain Randomization (DR):
        _env.domain_randomization = -1  # disable DR
        _env.observation_noise = -1  # disable DR
        # seeding
        seed = self.seed
        seed += 10000 * mpi.proc_id()
        np.random.seed(seed)
        _env.seed(seed)
        return _env


if __name__ == '__main__':
    of = ObjectiveFunctionCircleTask()
    x = of.sample()
    of.evaluate(x)
