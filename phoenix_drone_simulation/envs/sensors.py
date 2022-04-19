r""""Sensor functions.

Author:     Sven Gronauer

Updates:
    14.04.2022: Moved sensor noise from utils to sensor.py
"""
import pybullet as pb
import math
import abc
import numpy as np
from math import exp

from phoenix_drone_simulation.envs.utils import deg2rad


class SensorNoise:
    def __init__(
            self,
            pos_norm_std=0.002,
            pos_unif_range=0.001,
            vel_norm_std=0.01,
            vel_unif_range=0.,
            quat_norm_std=deg2rad(0.1),
            quat_unif_range=deg2rad(0.05),
            gyro_noise_density=0.000175,
            gyro_random_walk=0.0105,
            gyro_bias_correlation_time=1000.,
            gyro_turn_on_bias_sigma=deg2rad(5),
            bypass=False,
            acc_static_noise_std=0.002,
            acc_dynamic_noise_ratio=0.005
    ):
        """
        Source:
            A.Molchanov, T. Chen, W. Hönig, J. A.Preiss, N. Ayanian, G. S. Sukhatme
            University of Southern California
            Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors


        Args:
            pos_norm_std (float): std of pos gaus noise component
            pos_unif_range (float): range of pos unif noise component
            vel_norm_std (float): std of linear vel gaus noise component
            vel_unif_range (float): range of linear vel unif noise component
            quat_norm_std (float): std of rotational quaternion noisy angle gaus component
            quat_unif_range (float): range of rotational quaternion noisy angle gaus component
            gyro_gyro_noise_density: gyroscope noise, MPU-9250 spec
            gyro_random_walk: gyroscope noise, MPU-9250 spec
            gyro_bias_correlation_time: gyroscope noise, MPU-9250 spec
            # gyro_turn_on_bias_sigma: gyroscope noise, MPU-9250 spec (val 0.09)
            bypass: no noise
        """

        self.pos_norm_std = pos_norm_std
        self.pos_unif_range = pos_unif_range

        self.vel_norm_std = vel_norm_std
        self.vel_unif_range = vel_unif_range

        self.quat_norm_std = quat_norm_std
        self.quat_unif_range = quat_unif_range

        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.gyro_bias_correlation_time = gyro_bias_correlation_time
        self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
        self.gyro_bias = np.zeros(3)

        self.acc_static_noise_std = acc_static_noise_std
        self.acc_dynamic_noise_ratio = acc_dynamic_noise_ratio

        self.bypass = bypass

    def add_noise(self, pos, vel, rot, omega, acc, dt):
        if self.bypass:
            return pos, vel, rot, omega, acc

        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert omega.shape == (3,)

        # add noise to position measurement
        pos_offset = np.random.normal(loc=0., scale=self.pos_norm_std, size=3) + \
                    np.random.uniform(low=-self.pos_unif_range,
                                      high=self.pos_unif_range,
                                      size=3)
        noisy_pos = pos + pos_offset

        # add noise to linear velocity
        noisy_vel = vel + \
                    np.random.normal(loc=0., scale=self.vel_norm_std, size=3) + \
                    np.random.uniform(low=-self.vel_unif_range,
                                      high=self.vel_unif_range,
                                      size=3)

        # Noise in omega
        noisy_omega = self.add_noise_to_omega(omega, dt)

        # Noise in rotation
        theta = np.random.normal(0, self.quat_norm_std, size=3) + \
                np.random.uniform(-self.quat_unif_range, self.quat_unif_range,
                                  size=3)

        assert rot.shape == (3,), f'Expecting 3D vector for rotation.'
        # Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
        noisy_rot = np.clip(rot + theta,
                            a_min=[-np.pi, -np.pi / 2, -np.pi],
                            a_max=[np.pi, np.pi / 2, np.pi])
        # Accelerometer noise
        noisy_acc = acc + np.random.normal(loc=0.,
                                           scale=self.acc_static_noise_std,
                                           size=3) + \
                    acc * np.random.normal(loc=0.,
                                           scale=self.acc_dynamic_noise_ratio,
                                           size=3)

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    # copied from rotorS imu plugin
    def add_noise_to_omega(self, omega, dt):
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt ** 0.5)
        sigma_b_g_d = (-(sigma_g_d ** 2) * (
                self.gyro_bias_correlation_time / 2) * (
                               exp(-2 * dt / self.gyro_bias_correlation_time) - 1)) ** 0.5
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * np.random.normal(
            0, 1, 3)
        return omega + self.gyro_bias \
               + self.gyro_random_walk * np.random.normal(0, 1, 3) \
               + self.gyro_turn_on_bias_sigma * np.random.normal(0, 1, 3)




class Sensor(abc.ABC):
    """ Baseclass for sensor units."""
    def __init__(
            self,
            bc,
            offset: list,
            agent,
            obstacles: list,
            coordinate_system: int,
            rotate_with_agent: bool,
            visualize: bool
    ):
        self.agent = agent
        self.bc = bc
        self.coordinate_system = coordinate_system
        self.obstacles = obstacles
        self.offset = np.array(offset)
        self.rotate_with_agent = rotate_with_agent
        self.visualize = visualize  # indicates if rendering is enabled

    def get_observation(self) -> np.ndarray:
        """Synonym method for measure()."""
        return self.measure()

    @abc.abstractmethod
    def measure(self, *args, **kwargs) -> np.ndarray:
        """ Collect information about nearby and detectable objects/bodies."""
        raise NotImplementedError

    def set_offset(self, offset: list) -> None:
        """ By default, the sensor is placed at agent's root link position.
            However, sometimes it is useful to adjust the position by an offset.
        """
        assert np.array(offset).shape == (3, )
        self.offset = np.array(offset)

    # @property
    # def type(self):
    #     return self.__class__

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:
        """ Get the sensor dimension as vector."""
        raise NotImplementedError


class LIDARSensor(Sensor):
    """ A sensor that performs radial ray casts to collect intersection
        information about nearby obstacles.

        Note: until now, only the world frame coordination system is supported.

    """
    supported_frames = [pb.WORLD_FRAME, pb.LINK_FRAME]

    def __init__(
            self,
            bc,
            agent,
            number_rays,
            ray_length,
            obstacles,
            offset=(0, 0, 0),
            coordinate_system=pb.LINK_FRAME,
            hit_color=(0.95, 0.1, 0),
            miss_color=(0.25, 0.95, 0.05),
            rotate_with_agent=True,  # Rotate LIDAR rays with agent
            visualize=True
    ):

        assert pb.MAX_RAY_INTERSECTION_BATCH_SIZE > number_rays
        assert coordinate_system in LIDARSensor.supported_frames

        super().__init__(
            bc=bc,
            agent=agent,
            obstacles=obstacles,
            offset=offset,
            coordinate_system=coordinate_system,
            rotate_with_agent=rotate_with_agent,
            visualize=visualize)

        self.number_rays = number_rays
        self.ray_length = ray_length
        self.ray_width = 0.15

        # visualization parameters
        self.replace_lines = True
        self.hit_color = hit_color
        self.miss_color = miss_color

        # collect measurement information
        self.rays_starting_points = []
        self.rays_end_points = []
        self.ray_ids = self.init_rays()

    def init_rays(self) -> list:
        """ Spawn ray visuals in simulation and collect their body IDs.
            Note: Rays are spawned clock-wise beginning at 12 clock position.
        """
        from_position = np.array([0, 0, 0]) + self.offset
        ray_ids = []
        for i in range(self.number_rays):
            if self.replace_lines and self.visualize:
                end_point = [1, 1, 1]
                ray_ids.append(
                    self.bc.addUserDebugLine(from_position, end_point,
                                        self.miss_color, lineWidth=self.ray_width))
            else:
                ray_ids.append(-1)

        return ray_ids

    def set_ray_positions(self, from_position=None, shift=0.):
        if from_position is None:
            from_position = self.agent.get_position()
        assert from_position.shape == (3, ), f'Got shape={from_position.shape}'
        self.rays_starting_points = []
        self.rays_end_points = []

        if self.rotate_with_agent:
            abcd = self.agent.get_quaternion()
            R = np.array(self.bc.getMatrixFromQuaternion(abcd)).reshape((3, 3))
            start_pos = from_position + R @ self.offset
        else:
            start_pos = from_position + self.offset
        for i in range(self.number_rays):
            self.rays_starting_points.append(start_pos)
            angle = (2. * math.pi * float(i)) / self.number_rays + shift
            dx = self.ray_length * math.sin(angle)
            dy = self.ray_length * math.cos(angle)
            dz = 0.
            if self.rotate_with_agent:
                # Rotate LIDAR rays with agent
                rotated_delta_xyz = R @ np.array([dx, dy, dz])
                end_point = start_pos + rotated_delta_xyz
            else:
                # Keep rays parallel to the ground -> no rotation with agent
                end_point = start_pos + np.array([dx, dy, dz])
            self.rays_end_points.append(end_point)

    @property
    def shape(self) -> tuple:
        return (self.number_rays, )

    def render(self, data) -> None:
        """ Display and update ray visuals."""
        if not self.visualize:
            # Do not draw debug lines when visuals are not rendered
            return

        for i in range(self.number_rays):
            hitObjectUid = data[i][0]

            if hitObjectUid < 0:  # no object intersection
                # hitPosition = [0, 0, 0]
                self.bc.addUserDebugLine(
                    self.rays_starting_points[i],
                    self.rays_end_points[i],
                    self.miss_color,
                    lineWidth=self.ray_width,
                    replaceItemUniqueId=self.ray_ids[i])
            else:
                hitPosition = data[i][3]
                self.bc.addUserDebugLine(
                    self.rays_starting_points[i],
                    hitPosition,
                    self.hit_color,
                    lineWidth=self.ray_width ,
                    replaceItemUniqueId=self.ray_ids[i])

    def measure(self, from_position=None) -> np.ndarray:
        """
            origin_position: list holding 3 entries: [x, y, z]
        """
        self.set_ray_positions(from_position)  # if self.visualize else None
        # Detect distances to close bodies via ray casting (sent as batch)
        results = self.bc.rayTestBatch(
            self.rays_starting_points,
            self.rays_end_points,
            # parentObjectUniqueId=self.agent.body_id
        )
        if not self.replace_lines:
            self.bc.removeAllUserDebugItems()
        self.render(data=results)

        # distances to obstacle in range [0, 1]
        # 1: close to sensor
        # 0: not in reach of sensor
        distances = [1.0 - d[2] for d in results]

        return np.array(distances)
