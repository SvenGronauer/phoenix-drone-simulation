import abc
import numpy as np
from pybullet_utils import bullet_client

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

# ==============================================================================
#   Settings from the CrazyFlie Firmware, see:
#   https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/interface/pid.h
# ==============================================================================

PID_ROLL_RATE_KP = 250.0  # default: 250.0
PID_ROLL_RATE_KI = 500.0  # default: 500.0
PID_ROLL_RATE_KD = 2.5  # default: 2.50
PID_ROLL_RATE_INTEGRATION_LIMIT = 33.3

PID_PITCH_RATE_KP = 250.0  # default: 250.0
PID_PITCH_RATE_KI = 500.0  # default: 500.0
PID_PITCH_RATE_KD = 2.5  # default: 2.50
PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3

PID_YAW_RATE_KP = 120.0
PID_YAW_RATE_KI = 16.7
PID_YAW_RATE_KD = 0.0
PID_YAW_RATE_INTEGRATION_LIMIT = 166.7


def limitThrust(pwm_value):
    """Limit PWM signal."""
    return np.clip(pwm_value, 0, 60000)


def rpy_control_factors_to_PWM(rpy_control_factors: np.ndarray, 
                               thrust: float
                               ) -> np.ndarray:
    r"""Convert PID output to motor power signal (uint16).Assumes
     QUAD_FORMATION_X for CrazyFlie drone."""
    assert rpy_control_factors.shape == (3,)
    
    control_roll, control_pitch, control_yaw = rpy_control_factors
    r = control_roll / 2.0
    p = control_pitch / 2.0
    
    PWMs = np.empty(4)
    PWMs[0] = limitThrust(thrust - r - p - control_yaw)
    PWMs[1] = limitThrust(thrust - r + p + control_yaw)
    PWMs[2] = limitThrust(thrust + r + p - control_yaw)
    PWMs[3] = limitThrust(thrust + r - p + control_yaw)
    return PWMs


class Control(object):
    r"""Parent class for control objects."""

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,  # 1 / sim_frequency
    ):
        self.drone = drone
        self.bc = bc
        self.control_counter = 0
        self.time_step = time_step

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    @abc.abstractmethod
    def act(self, action, **kwargs):
        r"""Action to PWM signal."""
        raise NotImplementedError

    @classmethod
    def degree_to_rad(cls, x: np.ndarray) -> np.ndarray:
        return x / 180. * np.pi

    @classmethod
    def rad_to_degree(cls, x: np.ndarray) -> np.ndarray:
        return x * 180. / np.pi

    def reset(self):
        r"""Reset the control classes.

        A general use counter is set to zero.
        """
        self.control_counter = 0


class PWM(Control):
    r"""Class for direct PWM motor control."""

    def act(self, action, **kwargs):
        r"""Action to PWM signal. Reduced PWM range from [0, 65535] down to
        [0, 60000] in order to have ~ 8% left for battery compensation.
        """
        clipped_action = np.clip(action, -1, 1)
        PWMs = 30000 + clipped_action * 30000  # PWM in [0, 60000]
        return PWMs


# class PID(Control):
#     def __init__(
#             self,
#             drone,
#             bc: bullet_client.BulletClient,
#             time_step: float,
#             kp: float,
#             ki: float,
#             kd: float,
#     ):
#         super(PID, self).__init__(
#             drone=drone,
#             bc=bc,
#             time_step=time_step
#         )


class AttitudeRate(Control):
    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,  # 1 / sim_frequency
    ):
        super(AttitudeRate, self).__init__(
            drone=drone,
            bc=bc,
            time_step=time_step,
        )
        # self.state = None
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)

        # Attitude Rate parameters:
        self.kp_att_rate = np.array(
            [PID_ROLL_RATE_KP, PID_PITCH_RATE_KP, PID_YAW_RATE_KP])
        self.ki_att_rate = np.array(
            [PID_ROLL_RATE_KI, PID_PITCH_RATE_KI, PID_YAW_RATE_KI])
        self.kd_att_rate = np.array(
            [PID_ROLL_RATE_KD, PID_PITCH_RATE_KD, PID_YAW_RATE_KD])

        self.rpy_rate_integral_limits = np.array(
            [PID_ROLL_RATE_INTEGRATION_LIMIT,
             PID_PITCH_RATE_INTEGRATION_LIMIT,
             PID_YAW_RATE_INTEGRATION_LIMIT])

        self.reset()

    def act(self, action, **kwargs):
        """Action to PWM signal."""
        clipped_action = np.clip(action, -1, 1)
        # Action = [thrust, roll_dot, pitch_dot, yaw_dot]
        thrust = 30000 + clipped_action[0] * 30000
        rpy_dot_target = clipped_action[1:4] * np.pi/3
        rpy_factors = self.compute_output(rpy_dot_target)

        PWMs = rpy_control_factors_to_PWM(rpy_factors, thrust=thrust)
        return PWMs

    def compute_output(self,
                       rpy_dot_target: np.ndarray  # [rad/s] in local body frame
                       ) -> np.ndarray:
        """Computes the PID control action (as PWMs) for a single drone."""
        dt = self.time_step

        # Note: PyBullet calculates in rad whereas the firmware takes degrees
        error = self.rad_to_degree(rpy_dot_target - self.drone.rpy_dot)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        self.integral += error * dt
        # limit integral values
        self.integral = np.clip(self.integral, -self.rpy_rate_integral_limits,
                                self.rpy_rate_integral_limits)
        # print('self.integral:', self.integral)
        rpy_offsets = self.kp_att_rate * error \
                      + self.ki_att_rate * self.integral \
                      + self.kd_att_rate * derivative
        return rpy_offsets

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        # Initialized PID control variables
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)


class Attitude(Control):
    r"""Attitude controller.

    Parameters are taken from CrazyFlie Firmware:
    https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/interface/pid.h
    """
    PID_ROLL_KP = 6.0
    PID_ROLL_KI = 3.0
    PID_ROLL_KD = 0.0
    PID_ROLL_INTEGRATION_LIMIT = 20.0

    PID_PITCH_KP = 6.0
    PID_PITCH_KI = 3.0
    PID_PITCH_KD = 0.0
    PID_PITCH_INTEGRATION_LIMIT = 20.0

    PID_YAW_KP = 6.0
    PID_YAW_KI = 1.0
    PID_YAW_KD = 0.35
    PID_YAW_INTEGRATION_LIMIT = 360.0

    def __init__(
            self,
            drone,
            bc: bullet_client.BulletClient,
            time_step: float,  # 1 / sim_frequency
    ):
        super(Attitude, self).__init__(
            drone=drone,
            bc=bc,
            time_step=time_step,
        )
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)
        self.kps = np.array([Attitude.PID_ROLL_KP,
                             Attitude.PID_PITCH_KP,
                             Attitude.PID_YAW_KP])
        self.kis = np.array([Attitude.PID_ROLL_KI,
                             Attitude.PID_PITCH_KI,
                             Attitude.PID_YAW_KI])
        self.kds = np.array([Attitude.PID_ROLL_KD,
                             Attitude.PID_PITCH_KD,
                             Attitude.PID_YAW_KD])
        self.integral_limits = np.array([Attitude.PID_ROLL_INTEGRATION_LIMIT,
                                         Attitude.PID_PITCH_INTEGRATION_LIMIT,
                                         Attitude.PID_YAW_INTEGRATION_LIMIT])

        # Attitude control requires cascaded PID-controllers:
        self.attitude_rate_controller = AttitudeRate(drone=drone,
                                                     bc=bc,
                                                     time_step=time_step)
        self.reset()

    def act(self, action, **kwargs):
        """Action  [thrust, roll, pitch, yaw] to PWM signals.

        Calls Attitude Controller => Attitude Rate Controller => PWM
        """
        clipped_action = np.clip(action, -1, 1)
        rpy_target = clipped_action[1:4] * np.pi/18  # in rad

        # Action = [thrust, roll, pitch, yaw]
        thrust = 45000 + clipped_action[0] * 10000
        rpy_dot_targets = self.compute_output(rpy_target)  # in [rad / s]
        rpy_dot_factors = self.attitude_rate_controller.compute_output(
            rpy_dot_targets)
        PWMs = rpy_control_factors_to_PWM(rpy_dot_factors, thrust)

        return PWMs

    def compute_output(self,
                       rpy_target: np.ndarray  # [rad] in world coordinates
                       ) -> np.ndarray:  # in [rad / s]
        """Computes angle rates for the lower-level Attitude Rate Controller."""
        dt = self.time_step

        # Note: PyBullet calculates in rad whereas the firmware takes degrees
        error = self.rad_to_degree(rpy_target - self.drone.rpy)  # in [deg]
        derivative = (error - self.last_error) / dt
        self.last_error = error
        self.integral += error * dt
        self.integral = np.clip(self.integral,
                                -self.integral_limits,
                                self.integral_limits)
        rpy_offsets = self.kps * error + self.kis * self.integral \
                      + self.kds * derivative  # in [degree / s]
        return self.degree_to_rad(rpy_offsets)  # in [rad / s]

    def reset(self):
        r"""Resets the internal variables of the PIDs. """
        super().reset()
        self.attitude_rate_controller.reset()
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)
