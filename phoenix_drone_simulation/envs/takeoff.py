r"""Take-off task for CrazyFlie Drone (Project Phoenix)

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    19-08-2021
"""
import numpy as np

# local imports
from phoenix_drone_simulation.envs.base import DroneBaseEnv


class DroneTakeOffBaseEnv(DroneBaseEnv):
    def __init__(
            self,
            physics,
            control_mode: str,
            drone_model: str,
            observation_noise=1,  # must be positive in order to add noise
            domain_randomization: float = 0.10,  # use 10% DR as default
            target_pos: np.ndarray = np.array([0, 0, 1.0], dtype=np.float32),
            sim_freq=200,  # in Hz
            aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
            observation_frequency=100,  # in Hz
            penalty_action: float = 1e-4,
            penalty_angle: float = 0,
            penalty_spin: float = 1e-4,
            penalty_terminal: float = 100,
            penalty_velocity: float = 0,
            **kwargs
    ):
        # === Take-Off task specific attributes
        # must be defined before calling super class constructor:
        self.target_pos = target_pos  # used in _computePotential()
        self.ARP = 0
        self.penalty_action = penalty_action
        self.penalty_angle = penalty_angle
        self.penalty_spin = penalty_spin
        self.penalty_terminal = penalty_terminal
        self.penalty_velocity = penalty_velocity

        self.done_dist_threshold = 0.50  # in [m]

        # === Reference trajectory
        self.num_ref_points = N = 300
        self.ref = np.zeros((N, 3))
        self.ref[:N, 2] = np.arange(N) / N  # z-position
        self.ref[N:, 2] = 1.  # z-position
        self.ref_offset = 0  # set by task_specific_reset()-method

        # init drone state
        init_xyz = np.array([0, 0, 0.0125], dtype=np.float32)
        init_rpy = np.zeros(3)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        super(DroneTakeOffBaseEnv, self).__init__(
            control_mode=control_mode,
            drone_model=drone_model,
            init_xyz=init_xyz,
            init_rpy=init_rpy,
            init_xyz_dot=init_xyz_dot,
            init_rpy_dot=init_rpy_dot,
            physics=physics,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            sim_freq=sim_freq,
            aggregate_phy_steps=aggregate_phy_steps,
            observation_frequency=observation_frequency,
            **kwargs
        )

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.95, 0.1, 0.05, 0.4],
        )
        # Spawn visual without collision shape
        self.target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )

        # === Set camera position
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(-0.5, 0.5, 0.0),
            cameraDistance=2.,
            cameraYaw=45,
            cameraPitch=-45
        )

    def compute_done(self) -> bool:
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        done = True if dist > self.done_dist_threshold else False
        return False

    def compute_info(self) -> dict:
        c = 0.
        info = {'cost': c}
        return info

    def compute_observation(self) -> np.ndarray:
        t = int(min(self.iteration, self.num_ref_points-1))
        self.target_pos = self.ref[t]
        self.target_pos = self.ref[t]
        self.bc.resetBasePositionAndOrientation(
            self.target_body_id,
            posObj=self.target_pos,
            ornObj=(0, 0, 0, 1)
        )

        if self.observation_noise > 0:  # add noise only for positive values
            if self.iteration % self.obs_rate == 0:
                # update state information with 100 Hz (except for rpy_dot)
                # apply noise to perfect simulation state:
                xyz, vel, rpy, omega, acc = self.sensor_noise.add_noise(
                    pos=self.drone.xyz,
                    vel=self.drone.xyz_dot,
                    rot=self.drone.rpy,
                    omega=self.drone.rpy_dot,
                    acc=np.zeros(3),  # irrelevant
                    dt=1/self.SIM_FREQ
                )
                quat = np.asarray(self.bc.getQuaternionFromEuler(rpy))
                error_to_ref = self.target_pos - xyz
                self.state = np.concatenate(
                    [xyz, quat, vel, omega, self.drone.last_action])
            else:
                # This part runs with >100Hz, re-use Kalman Filter values:
                xyz, quat, vel = self.state[0:3], self.state[3:7], self.state[7:10]
                error_to_ref = self.ref[t] - xyz
                # read Gyro data with >100 Hz and add noise:
                omega = self.sensor_noise.add_noise_to_omega(
                    omega=self.drone.rpy_dot, dt=1/self.SIM_FREQ)

            # apply low-pass filtering to gyro
            omega = self.gyro_lpf.apply(omega)
            obs = np.concatenate(
                [xyz, quat, vel, omega, self.drone.last_action, error_to_ref])
        else:
            # no observation noise is applied
            error_to_ref = self.target_pos - self.drone.xyz
            obs = np.concatenate([self.drone.get_state(), error_to_ref])
        return obs

    def compute_potential(self) -> float:
        """Euclidean distance from current ron position to target position."""
        return np.linalg.norm(self.drone.xyz - self.target_pos)

    def compute_reward(self, action) -> float:
        r"""Calculates the reward."""
        act_diff = action - self.drone.last_action
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)

        penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)
        penalty_action_rate = self.ARP * np.linalg.norm(act_diff)
        penalty_rpy = self.penalty_angle * np.linalg.norm(self.drone.rpy)
        penalty_spin = self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        penalty_terminal = self.penalty_terminal if self.compute_done() else 0.
        penalty_velocity = self.penalty_action * np.linalg.norm(self.drone.xyz_dot)

        penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_spin,
                            penalty_velocity, penalty_action, penalty_terminal])
        # L2 norm
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        reward = -dist - penalties
        if self.drone.xyz[2] < 0.08:
            reward -= 1.
        return reward

    def get_reference_trajectory(self):
        raise NotImplementedError

    def task_specific_reset(self):
        # set random offset for position
        pos = self.init_xyz.copy()  # pos[:2] is call by reference, so copy
        quat = self.init_quaternion

        if self.enable_reset_distribution:
            # Note: Only the XY Position is set, Z is set such that drone
            #  touches the ground.
            xy_lim = 0.25
            pos[:2] += np.random.uniform(-xy_lim, xy_lim, size=2)

            # sample yaw from [-pi,+pi]
            rpy = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            quat = self.bc.getQuaternionFromEuler(rpy)

        self.old_potential = pos
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )
        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=self.init_xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ self.init_rpy_dot
        )

        # Set drone internals accordingly
        self.drone.x[:] = 0.
        self.drone.y = self.drone.K * self.drone.x
        self.drone.action_buffer[:] = -1  # equals 0% PWM
        self.drone.last_action[:] = -1  # equals 0% PWM


""" ==================  
        PWM control
    ==================
"""


class DroneTakeOffSimpleEnv(DroneTakeOffBaseEnv):
    def __init__(self, **kwargs):
        super(DroneTakeOffSimpleEnv, self).__init__(
            aggregate_phy_steps=1,
            control_mode='PWM',
            drone_model='cf21x_sys_eq',
            physics='SimplePhysics',
            # use 100 Hz since no motor dynamics and PID is used
            sim_freq=100,
            **kwargs
        )


class DroneTakeOffBulletEnv(DroneTakeOffBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 2,  # sub-steps used to calculate motor dynamics
                 control_mode: str = 'PWM',
                 **kwargs):
        super(DroneTakeOffBulletEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )
