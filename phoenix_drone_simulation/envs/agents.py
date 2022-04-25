r""""Agent classes for Project Phoenix.

Author:     Sven Gronauer
Created:    14.04.2021
Updates:    17.05.2021
"""
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client
import abc
import os
import xml.etree.ElementTree as etxml
from typing import Tuple
import phoenix_drone_simulation.envs.control as phoenix_control
from phoenix_drone_simulation.envs.utils import get_assets_path, OUNoise
from typing import Optional


class AgentBase(abc.ABC):
    r"""Base class for agents."""
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            name: str,
            file_name: str,
            act_dim: int,
            obs_dim: int,
            time_step: float,
            aggregate_phy_steps: int,
            color: tuple = (1., 1., 1, 1.0),
            xyz: np.ndarray = np.array([0., 0., 1.], dtype=np.float64),  # [m]
            rpy: np.ndarray = np.zeros(3, dtype=np.float64),   # [rad]
            xyz_dot: np.ndarray = np.zeros(3, dtype=np.float64),  # [m/s]
            rpy_dot: np.ndarray = np.zeros(3, dtype=np.float64),  # [rad/s]
            fixed_base=False,
            global_scaling=1,
            self_collision=False,
            verbose=False,
            debug=False,
            **kwargs
    ):
        assert len(rpy) == 3, 'init_orientation expects (r,p,y)'
        assert len(xyz) == 3
        self.aggregate_phy_steps = aggregate_phy_steps
        self.bc = bc
        self.name = name
        self.file_name = file_name
        self.fixed_base = 1 if fixed_base else 0
        self.file_name_path = os.path.join(get_assets_path(), self.file_name)
        self.global_scaling = global_scaling
        self.xyz = xyz   # [m]
        self.xyz_dot = xyz_dot   # [m/s] world coordinates
        self.rpy = rpy   # [rad]
        self.quaternion = np.array(pb.getQuaternionFromEuler(rpy))
        self.rpy_dot = rpy_dot   # [rad/s] local body frame
        self.color = np.array(color)
        self.self_collision = self_collision
        self.visible = True
        self.verbose = verbose
        self.time_step = time_step
        self.debug = debug

        # space information
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.body_unique_id = self.load_assets()

        # Setup controller
        assert hasattr(phoenix_control, control_mode), \
            f'Control={control_mode} not found.'
        control_cls = getattr(phoenix_control, control_mode)  # get class reference
        self.control = control_cls(
            self,  # pass Drone class
            self.bc,
            time_step=time_step,  # 1 / sim_frequency
        )

    @abc.abstractmethod
    def apply_force(self,
                    force):
        """Apply force vector to drone motors."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_z_torque(self,
                       force):
        """Apply torque responsible for yaw."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self
                  ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def load_assets(self) -> int:
        """Loads the robot description file into the simulation."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self
              ) -> None:
        """Agent specific reset."""
        raise NotImplementedError

    def violates_constraints(self, does_violate_constraint):
        """Displays a red sphere which indicates the receiving of costs when
        enable is True, else deactivate visual shape."""
        pass


class CrazyFlieAgent(AgentBase):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            file_name: str,   # two URDF files are implemented: cf21x_bullet.urdf and cf21x_sys_eq.urdf
            time_step: float,
            aggregate_phy_steps: int,
            latency: float,  # [s]
            motor_time_constant: float,  # [s]
            motor_thrust_noise: float,  # [1] noise in % added to thrusts
            use_latency: bool = True,
            use_motor_dynamics: bool = True,
            display_local_frame: bool = True
    ):
        super(CrazyFlieAgent, self).__init__(
            bc=bc,
            control_mode=control_mode,
            name='CrazyFlie2.1X',
            file_name=file_name,
            act_dim=4,
            obs_dim=22,  # New: increased from 16 to 22 obs dim
            time_step=time_step,
            aggregate_phy_steps=aggregate_phy_steps,
        )
        self._parse_robot_parameters()

        # Parameters from Julian Förster:
        self.FORCE_TORQUE_FACTOR_0 = 1.56e-5
        self.FORCE_TORQUE_FACTOR_1 = 5.96e-3

        self.G = 9.81  # m s^-2

        # Compute constants
        self.GRAVITY = self.G * self.M
        self.MAX_THRUST = self.GRAVITY * self.THRUST2WEIGHT_RATIO / 4
        self.MAX_TORQUE = self.FORCE_TORQUE_FACTOR_1 * self.MAX_THRUST

        self.HOVER_X = np.sqrt(1/self.THRUST2WEIGHT_RATIO)
        self.HOVER_ACTION = 2 * 1/self.THRUST2WEIGHT_RATIO - 1
        # Note: self.MAX_RPM is equal to 1 since no KF factor in here..
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.MAX_THRUST))
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

        self.TIME_STEP = time_step

        # parameters changed by domain randomization
        self.m = self.M  # mass of drone in [kg]
        self.kf = self.KF
        self.km = self.KM  # set to 7.94e-12 in URDF file
        self.thrust_to_weight_ratio = self.THRUST2WEIGHT_RATIO
        self.use_latency = use_latency if latency >= time_step else False

        self.force_torque_factor_0 = self.FORCE_TORQUE_FACTOR_0
        self.force_torque_factor_1 = self.FORCE_TORQUE_FACTOR_1

        self.PWMs = np.zeros(4)
        self.last_action = np.zeros(self.act_dim)

        # === setup motor dynamics (PT1 behavior)
        self.LATENCY = latency  # delay of system in [s]
        # Fixed motor constants used for drawing new values in domain rand.:
        self.MOTOR_TIME_CONSTANT = motor_time_constant  # in [s]
        self.MOTOR_THRUST_NOISE = motor_thrust_noise

        self.control_loop_freq = int(1./(time_step*aggregate_phy_steps))  # [Hz]
        self.buf_size = int(max(1, int(self.LATENCY // time_step)))
        assert self.buf_size > 0
        self.action_buffer = np.zeros(shape=(self.buf_size, self.act_dim))
        self.action_idx = 0

        self.x = np.zeros(self.act_dim)
        self.y = np.zeros(self.act_dim)

        # GUI variables
        self.DISPLAY_LOCAL_FRAME = display_local_frame
        self.axis_x = -1
        self.axis_y = -1
        self.axis_z = -1

        # ===> Important note:
        #   the following parameters are manipulated by domain randomization
        self.use_motor_dynamics = use_motor_dynamics
        self.T = self.MOTOR_TIME_CONSTANT
        self.latency = self.LATENCY   # system latency in [s]
        self.T_s = self.time_step  # 1 / SIM_FREQ; usually 0.002
        self.K = self.MAX_THRUST   # max thrust per motor when PWM == 65535

        # Note that A, B, K are manipulated by domain randomization
        self.A = np.ones(self.act_dim) * (1 - self.T_s / self.T)
        self.B = np.ones(self.act_dim) * self.T_s / self.T
        # sigma = 0.2 gives roughly max noise of -1 .. 1
        self.thrust_noise = OUNoise(self.act_dim, sigma=0.2*self.MOTOR_THRUST_NOISE)

    def update_motor_dynamics(
            self,
            new_motor_time_constant: Optional[np.ndarray] = None,
            new_sampling_time: Optional[float] = None,
            new_thrust_to_weight_ratio: Optional[np.ndarray] = None
    ) -> None:
        """Build discrete system description of motor behavior."""
        if new_sampling_time is not None:
            self.T_s = new_sampling_time
        if new_motor_time_constant is not None:
            self.T = np.clip(new_motor_time_constant, self.T_s, np.inf)
        if new_thrust_to_weight_ratio is not None:
            self.thrust_to_weight_ratio = new_thrust_to_weight_ratio

        self.A = 1 - self.T_s / self.T
        self.B = self.T_s / self.T
        self.K = 0.028 * self.G * self.thrust_to_weight_ratio / 4

    def _parse_robot_parameters(self
                                ) -> None:
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        URDF_TREE = etxml.parse(self.file_name_path).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        # self.THRUST2WEIGHT_RATIO = 2.5
        self.IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        self.IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        self.IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([self.IXX, self.IYY, self.IZZ])
        self.J_INV = np.linalg.inv(self.J)
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
        self.COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        self.COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        self.COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        self.COLLISION_Z_OFFSET = self.COLLISION_SHAPE_OFFSETS[2]
        self.MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        self.GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        self.PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        self.DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])  # [kg /rad]
        self.DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])  # [kg /rad]
        self.DRAG_COEFF = np.array([self.DRAG_COEFF_XY, self.DRAG_COEFF_XY, self.DRAG_COEFF_Z])  # [kg /rad]
        self.DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        self.DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        self.DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])

    def apply_action(self,
                     action
                     ) -> Tuple[np.ndarray, float]:
        """Returns the forces that are applied to drone motors."""

        self.last_action = action.copy()
        # action to PWM signal based on the control mode (PWM, Attitude Rate)
        # Note: clipping of action is done in act() method of control
        if self.use_latency:
            # called with 500 Hz (=2ms): Target delay is 20ms: cycle of 10 steps
            # get delayed action first
            delayed_action = self.action_buffer[self.action_idx].copy()
            # then set current action for later
            self.action_buffer[self.action_idx] = action
            self.action_idx = (self.action_idx + 1) % self.buf_size
            PWMs = self.PWMs = self.control.act(action=delayed_action)
        else:
            PWMs = self.PWMs = self.control.act(action=action)

        thrust_noise = self.thrust_noise.noise()

        thrust_normed = PWMs / 60000  # cast into [0, 1]
        # ==== First-order motor dynamics (as stated in Molchanov et al.) ===
        # x(k+1) = A x(k) + B u(k)
        # y(k+1) = K * x(k+1)
        if self.use_motor_dynamics:
            # NN commands thrusts: so we need to convert to rot vel and back
            rot_normed = np.sqrt(thrust_normed)
            self.x = self.A * self.x + self.B * rot_normed
            noisy_x = (1 + thrust_noise) * self.x ** 2
        else:
            noisy_x = (1 + thrust_noise) * thrust_normed
        n = np.clip(noisy_x, 0, 1)
        self.y = self.K * n  # self.angvel2thrust(n)
        current_motor_forces = self.y

        torques = self.force_torque_factor_1 * current_motor_forces \
                  + self.force_torque_factor_0
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        return current_motor_forces, z_torque

    def apply_force(self, force, frame=pb.LINK_FRAME):
        """Apply a force vector to mass center of drone."""
        assert force.size == 3
        self.bc.applyExternalForce(
            self.body_unique_id,
            4,  # center of mass link
            forceObj=force,
            posObj=[0, 0, 0],
            flags=frame
        )

    def apply_motor_forces(self, forces):
        """Apply a force vector to the drone motors."""
        assert forces.size == 4
        for i in range(4):
            self.bc.applyExternalForce(
                self.body_unique_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=pb.LINK_FRAME
            )
        # Do motor speed visualization
        for i in range(4):
            self.bc.setJointMotorControl2(bodyUniqueId=self.body_unique_id,
                                    jointIndex=i,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    targetVelocity=self.x[i]*100,
                                    force=0.010)

    def apply_z_torque(self, torque):
        """Apply torque responsible for yaw."""
        self.bc.applyExternalTorque(
            self.body_unique_id,
            4,  # center of mass link
            torqueObj=[0, 0, torque],
            flags=pb.LINK_FRAME
        )

    def get_state(self
                  ) -> np.ndarray:
        state = np.concatenate([
            self.xyz,
            self.quaternion,
            self.xyz_dot,
            self.rpy_dot,  # local body frame
            self.last_action
        ])
        return state.reshape(17, )

    def load_assets(self
                    ) -> int:
        """Loads the robot description file into the simulation.

        Expected file format: URDF

        Returns
        -------
            body_unique_id of loaded body
        """
        assert self.file_name_path.endswith('.urdf')
        assert os.path.exists(self.file_name_path), \
            f'Did not find {self.file_name} at: {get_assets_path()}'
        # print(f'file_name_path: {self.file_name_path} ')
        random_xyz = (0, 0, 0)
        random_rpy = (0, 0, 0)

        body_unique_id = self.bc.loadURDF(
            self.file_name_path,
            random_xyz,
            pb.getQuaternionFromEuler(random_rpy),
            # Important Note: take inertia from URDF...
            flags=pb.URDF_USE_INERTIA_FROM_FILE
        )
        assert body_unique_id >= 0  # , msg
        return body_unique_id

    def reset(self
              ) -> None:
        """Agent specific reset function."""
        self.control.reset()   # reset PID control or PWM control
        # Note: the following values can be over-written by task_specific_reset
        self.x = np.zeros(self.act_dim)
        self.y = np.zeros(self.act_dim)
        self.action_idx = 0
        self.action_buffer = np.zeros_like(self.action_buffer)
        self.last_action = self.action_buffer[-1, :]

    def set_latency(self,
                    new_latency: float
                    ) -> None:
        r"""Update system latency: called by SimOpt methods.

        Remark: Latency is disabled when value is smaller than simulation time
        step.
        """
        self.latency = new_latency
        if new_latency < self.TIME_STEP:
            self.use_latency = False
        else:
            self.use_latency = True
            self.buf_size = int(self.latency / self.TIME_STEP)
            assert self.buf_size > 0
            self.action_buffer = np.zeros(shape=(self.buf_size, self.act_dim))
            self.action_idx = 0

    def show_local_frame(self):
        if self.DISPLAY_LOCAL_FRAME:
            AXIS_LENGTH = 2*self.L
            self.axis_x = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_x
                )
            self.axis_y = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_y,
                )
            self.axis_z = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_z
                )

    def update_information(self) -> None:
        r""""Retrieve drone's kinematic information from PyBullet simulation.

            xyz:        [m] in cartesian world coordinates
            rpy:        [rad] in cartesian world coordinates
            xyz_dot:    [m/s] in cartesian world coordinates
            rpy_dot:    [rad/s] in body frame
        """
        bid = self.body_unique_id
        pos, quat = self.bc.getBasePositionAndOrientation(bid)
        self.xyz = np.array(pos, dtype=np.float64)  # [m] in cartesian world coordinates
        self.quaternion = np.array(quat, dtype=np.float64)  # orientation as quaternion
        self.rpy = np.array(self.bc.getEulerFromQuaternion(quat), dtype=np.float64)  # [rad] in cartesian world coordinates

        # PyBullet returns velocities of base in Cartesian world coordinates
        xyz_dot_world, rpy_dot_world = self.bc.getBaseVelocity(bid)
        self.xyz_dot = np.array(xyz_dot_world, dtype=np.float64)  # [m/s] in world frame
        # FIXED: transform omega from world frame to local drone frame
        R = np.asarray(self.bc.getMatrixFromQuaternion(quat)).reshape((3, 3))
        self.rpy_dot = R.T @ np.array(rpy_dot_world, dtype=np.float64)  # [rad/s] in body frame


class CrazyFlieBulletAgent(CrazyFlieAgent):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            time_step: float,
            aggregate_phy_steps: int,
            **kwargs
    ):
        super().__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            bc=bc,
            control_mode=control_mode,
            file_name='cf21x_bullet.urdf',
            time_step=time_step,
            use_latency=True,
            use_motor_dynamics=True,  # use first-order motor dynamics
            **kwargs
        )


class CrazyFlieSimpleAgent(CrazyFlieAgent):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            time_step: float,
            aggregate_phy_steps: int,
            **kwargs
    ):
        super(CrazyFlieSimpleAgent, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            bc=bc,
            control_mode=control_mode,
            file_name='cf21x_sys_eq.urdf',
            time_step=time_step,
            use_latency=False,
            use_motor_dynamics=False,  # disable first-order motor dynamics
            **kwargs
        )
