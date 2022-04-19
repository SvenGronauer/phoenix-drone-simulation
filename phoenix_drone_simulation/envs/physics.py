import abc
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client
from typing import Tuple


class BasePhysics(abc.ABC):
    """Parent class."""

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,
            gravity: float = 9.81,
            number_solver_iterations: int = 5,
            use_ground_effect: bool = False
    ):
        self.drone = drone
        self.bc = bc
        self.time_step = time_step
        self.G = gravity
        self.number_solver_iterations = number_solver_iterations
        self.use_ground_effect = use_ground_effect

    def calculate_ground_effect(
            self,
            motor_forces: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        r"""Implementation of a ground effect model.

        Taken from:
        https://github.com/utiasDSL/gym-pybullet-drones/blob/a133c163e533ef1f5c55d7c1c631653e17f3bd79/gym_pybullet_drones/envs/BaseAviary.py#L709
        """
        # Kinematic info of all links (propellers and center of mass)
        link_states = self.bc.getLinkStates(
            self.drone.body_unique_id,
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        gec, r = self.drone.GND_EFF_COEFF, self.drone.PROP_RADIUS
        roll, pitch = self.drone.rpy[0:2]

        prop_z = np.array([
            link_states[0][0][2],  # z-position of link 0
            link_states[1][0][2],  # z-position of link 1
            link_states[2][0][2],  # z-position of link 2
            link_states[3][0][2]  # z-position of link 3
        ])
        prop_z = np.clip(prop_z, self.drone.GND_EFF_H_CLIP, np.inf)
        # Simple, per-propeller ground effects
        gnd_effects = motor_forces * gec * (r/(4 * prop_z))**2
        if np.abs(roll) < np.pi/2 and np.abs(pitch) < np.pi/2:
            return True, gnd_effects
        else:
            return False, np.zeros_like(gnd_effects)

    def set_parameters(
            self,
            time_step: float,
            number_solver_iterations: int,
            # **kwargs
    ):
        self.time_step = time_step
        self.number_solver_iterations = number_solver_iterations

    @abc.abstractmethod
    def step_forward(self,
                     action: np.ndarray,
                     *args,
                     **kwargs
                     ) -> None:
        r"""Steps the physics once forward."""
        raise NotImplementedError


class PyBulletPhysics(BasePhysics):
    
    def set_parameters(self, *args, **kwargs):
        super(PyBulletPhysics, self).set_parameters(*args, **kwargs)
        # Update PyBullet Physics
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1
        )

    def step_forward(self, action, *args, **kwargs):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        action
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        motor_forces, z_torque = self.drone.apply_action(action)

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.drone.apply_motor_forces(motor_forces)
        self.drone.apply_z_torque(z_torque)

        # === add drag effect
        quat = self.drone.quaternion
        vel = self.drone.xyz_dot
        base_rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Simple draft model applied to the base/center of mass
        rpm = self.drone.x**2 * 25000
        drag_factors = -1 * self.drone.DRAG_COEFF * np.sum(2*np.pi*rpm/60)
        drag = np.dot(base_rot, drag_factors*np.array(vel))
        # print(f'Drag: {drag}')
        self.drone.apply_force(force=drag)

        # === Ground Effect
        apply_ground_eff, ge_forces = self.calculate_ground_effect(motor_forces)
        if apply_ground_eff and self.use_ground_effect:
            self.drone.apply_motor_forces(forces=ge_forces)

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()


class SimplePhysics(BasePhysics):
    """Simplified version of system difference equations."""

    def step_forward(
            self,
            action: np.ndarray,
            *args,
            **kwargs
    ) -> None:
        """Explicit but simplified model dynamics implementation.

            xyz:        [m] in cartesian world coordinates
            rpy:        [rad] in cartesian world coordinates
            xyz_dot:    [m/s] in cartesian world coordinates
            rpy_dot:    [rad/s] in body frame

        Parameters
        ----------
        action:
            action computed by agent policy
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        forces, z_torque = self.drone.apply_action(action)

        # Retrieve current state from agent: (copy to avoid call by reference)
        pos = self.drone.xyz.copy()
        quat = self.drone.quaternion.copy()
        rpy = self.drone.rpy.copy()
        vel = self.drone.xyz_dot.copy()
        rpy_dot = self.drone.rpy_dot.copy()

        # transform thrust from body frame to world frame
        thrust = np.array([0, 0, np.sum(forces)])
        R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        thrust_world_frame = np.dot(R, thrust)
        force_world_frame = thrust_world_frame - np.array(
            [0, 0, self.G]) * self.drone.m

        # Note: based on X-configuration of Drone with arm length L=3.97cm
        # such that motors are positioned at (+-0.028, +-0.028)
        x_torque = (-forces[0] - forces[1] + forces[2] + forces[3]) * self.drone.L / np.sqrt(2)
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * self.drone.L / np.sqrt(2)

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_dot, np.dot(self.drone.J, rpy_dot))
        rpy_dot_dot = np.dot(self.drone.J_INV, torques)
        acc_linear = force_world_frame / self.drone.m

        vel += self.time_step * acc_linear
        rpy_dot += self.time_step * rpy_dot_dot
        pos += self.time_step * vel
        rpy += self.time_step * rpy_dot
        quaternion = np.array(self.bc.getQuaternionFromEuler(rpy))

        # Clipping z value to positive value (assumes hitting the ground plane)
        pos[2] = np.clip(pos[2], 0, np.inf)

        # Update drone internals and set new state information to PyBullet
        self.drone.xyz = pos  # [m] in cartesian world coordinates
        self.drone.quaternion = quaternion
        self.drone.rpy = rpy  # [rad] in cartesian world coordinates
        self.drone.xyz_dot = vel  # [m/s] in cartesian world coordinates
        self.drone.rpy_dot = rpy_dot  # [rad/s] in body frame
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            pos,
            quaternion
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            vel,
            # PyBullet assumes world frame: local frame -> world frame
            R.T @ rpy_dot
        )
