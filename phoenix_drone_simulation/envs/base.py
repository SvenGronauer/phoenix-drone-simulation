import os

import numpy as np
import pybullet as pb
import pybullet_data
import gym
from pybullet_utils import bullet_client
import abc

# local imports:
import phoenix_drone_simulation.envs.physics as phoenix_physics
from phoenix_drone_simulation.envs.agents import CrazyFlieSimpleAgent, \
    CrazyFlieBulletAgent
from phoenix_drone_simulation.envs.utils import get_assets_path
from phoenix_drone_simulation.envs.sensors import SensorNoise
from phoenix_drone_simulation.envs.utils import LowPassFilter, get_quaternion_from_euler
from collections import deque


class DroneBaseEnv(gym.Env, abc.ABC):
    """Base class for all drone environments."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            physics: str,  # e.g. PyBulletPhysics or SimplifiedSimplePhysics
            control_mode: str,  # use one of: PWM, Attitude, AttitudeRate
            drone_model: str,  # options: [cf21x_bullet, cf21x_sys_eq]
            init_xyz: np.ndarray,
            init_rpy: np.ndarray,
            init_xyz_dot: np.ndarray,
            init_rpy_dot: np.ndarray,
            aggregate_phy_steps: int = 2,
            debug=False,
            domain_randomization: float = -1,  # deactivated when negative value
            enable_reset_distribution=True,
            graphics=False,
            latency: float = 0.015,  # [s]
            motor_time_constant: float = 0.080,  # [s]
            motor_thrust_noise: float = 0.05,  # noise in % added to thrusts
            observation_frequency: int = 100,
            observation_history_size: int = 2,
            observation_noise=0.0,  # default: no noise added to obs
            sim_freq: int = 200
    ):
        """

        ::Notes::
        - Domain Randomization (DR) is applied when calling reset method and the
            domain_randomization is a positive float.

        Parameters
        ----------
        physics: str
            Name of physics class to be instantiated.
        drone_model
        init_xyz
        init_rpy
        init_xyz_dot
        init_rpy_dot
        sim_freq
        aggregate_phy_steps
        domain_randomization:
            Apply domain randomization to system parameters if value > 0
        graphics
        debug

        Raises
        ------
        AssertionError
            If no class is found for given physics string.
        """
        self.debug = debug
        self.domain_randomization = domain_randomization
        self.drone_model = drone_model
        self.enable_reset_distribution = enable_reset_distribution
        self.input_parameters = locals()  # save setting for later reset
        self.use_graphics = graphics

        # init drone state
        assert np.ndim(init_xyz) == 1 and np.ndim(init_rpy) == 1
        assert np.ndim(init_xyz_dot) == 1 and np.ndim(init_rpy_dot) == 1
        self.init_xyz = init_xyz
        self.init_rpy = init_rpy
        self.init_quaternion = get_quaternion_from_euler(init_rpy)
        self.init_xyz_dot = init_xyz_dot
        self.init_rpy_dot = init_rpy_dot

        # Default simulation constants (in capital letters)
        self.G = 9.8
        self.RAD_TO_DEG = 180 / np.pi
        self.DEG_TO_RAD = np.pi / 180
        self.SIM_FREQ = sim_freq  # default: 200Hz
        self.TIME_STEP = 1. / self.SIM_FREQ  # default: 0.002

        # Physics parameters depend on the task
        self.time_step = self.TIME_STEP
        self.number_solver_iterations = 5
        self.aggregate_phy_steps = aggregate_phy_steps
        assert aggregate_phy_steps >= 1

        # === Setup sensor and observation settings
        self.observation_frequency = observation_frequency
        self.obs_rate = int(sim_freq // observation_frequency)
        self.gyro_lpf = LowPassFilter(gain=1., time_constant=2/sim_freq,
                                      sample_time=1/sim_freq)

        self.iteration = 0

        # === Initialize and setup PyBullet ===
        self.bc = self._setup_client_and_physics(self.use_graphics)
        self.stored_state_id = -1

        # === spawn plane and drone agent ===
        self.agent_params = dict(
            aggregate_phy_steps=self.aggregate_phy_steps,
            control_mode=control_mode,
            latency=latency,
            motor_time_constant=motor_time_constant,
            motor_thrust_noise=motor_thrust_noise,
            time_step=self.time_step,
        )
        self._setup_simulation(physics=physics)

        self.state = np.zeros(17)
        use_observation_noise = observation_noise > 0
        self.sensor_noise = SensorNoise(bypass=not use_observation_noise)

        # === Observation space and action space ===
        # History of obs and actions
        assert observation_history_size >= 1
        self.observation_history = deque(maxlen=observation_history_size)
        self.action_history = deque(maxlen=observation_history_size)

        # negative noise values denote that zero noise is applied
        self.observation_noise = observation_noise
        self.observation_history_size = H = observation_history_size
        act_dim = self.drone.act_dim
        obs_dim = H * (self.compute_observation().size + act_dim)
        self.last_action = np.zeros(act_dim)

        # Define limits for observation space and action space
        o_lim = 1000 * np.ones((obs_dim, ), dtype=np.float32)
        a_lim = np.ones((act_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-o_lim, o_lim, dtype=np.float32)
        self.action_space = gym.spaces.Box(-a_lim, a_lim, dtype=np.float32)

        # stepping information
        self.old_potential = self.compute_potential()

    def _setup_client_and_physics(
            self,
            graphics=False
    ) -> bullet_client.BulletClient:
        r"""Creates a PyBullet process instance.

        The parameters for the physics simulation are determined by the
        get_physics_parameters() function.

        Parameters
        ----------
        graphics: bool
            If True PyBullet shows graphical user interface with 3D OpenGL
            rendering.

        Returns
        -------
        bc: BulletClient
            The instance of the created PyBullet client process.
        """
        if graphics or self.use_graphics:
            bc = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)

        # add open_safety_gym/envs/data to the PyBullet data path
        bc.setAdditionalSearchPath(get_assets_path())
        # disable GUI debug visuals
        bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1)
        bc.setGravity(0, 0, -9.81)
        # bc.setDefaultContactERP(0.9)
        return bc

    def _setup_simulation(
            self,
            physics: str,
    ) -> None:
        r"""Create world layout, spawn agent and obstacles.

        Takes the passed parameters from the class instantiation: __init__().
        """
        # reset some variables that might be changed by DR -- this avoids errors
        # when calling the render() method after training.
        self.g = self.G
        self.time_step = self.TIME_STEP

        # also add PyBullet's data path
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = self.bc.loadURDF("plane.urdf")
        # Load 10x10 Walls
        pb.loadURDF(os.path.join(get_assets_path(), "room_10x10.urdf"), useFixedBase=True)
        # random spawns

        if self.drone_model == 'cf21x_bullet':
            self.drone = CrazyFlieBulletAgent(bc=self.bc, **self.agent_params)
        elif self.drone_model == 'cf21x_sys_eq':
            self.drone = CrazyFlieSimpleAgent(bc=self.bc, **self.agent_params)
        else:
            raise NotImplementedError

        # Setup forward dynamics - Instantiates a particular physics class.
        assert hasattr(phoenix_physics, physics), \
            f'Physics={physics} not found.'
        physics_cls = getattr(phoenix_physics, physics)  # get class reference
        # call class constructor
        self.physics = physics_cls(
            self.drone,
            self.bc,
            time_step=self.time_step,  # 1 / sim_frequency
        )

        # Setup task specifics
        self._setup_task_specifics()

    @abc.abstractmethod
    def _setup_task_specifics(self):
        raise NotImplementedError

    def apply_domain_randomization(self) -> None:
        """ Apply domain randomization at the start of every new episode.

        Initialize simulation constants used for domain randomization. The
        following values are reset at the beginning of every epoch:
            - physics time step
            - Thrust-to-weight ratio
            - quadrotor mass
            - diagonal of inertia matrix
            - motor time constant
            - yaw-torque factors km1 and km2
        """
        def drawn_new_value(default_value,
                            factor=self.domain_randomization,
                            size=None):
            """Draw a random value from a uniform distribution."""
            bound = factor * default_value
            bounds = (default_value - bound, default_value + bound)
            return np.random.uniform(*bounds, size=size)

        if self.domain_randomization > 0:
            # physics parameter
            self.time_step = drawn_new_value(self.TIME_STEP)
            self.physics.set_parameters(
                time_step=self.time_step,
                number_solver_iterations=self.number_solver_iterations,
            )

            # === Drone parameters ====
            self.drone.m = drawn_new_value(self.drone.M)
            J_diag = np.array([self.drone.IXX, self.drone.IYY, self.drone.IZZ])
            J_diag_sampled = drawn_new_value(J_diag, size=3)
            self.drone.J = np.diag(J_diag_sampled)
            self.drone.J_INV = np.linalg.inv(self.drone.J)

            self.drone.force_torque_factor_0 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_0)
            self.drone.force_torque_factor_1 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_1)

            if self.drone.use_motor_dynamics:
                # set A, B, K according to new values of T_s, T
                mtc = drawn_new_value(self.drone.MOTOR_TIME_CONSTANT, size=4)
                t2w = drawn_new_value(self.drone.THRUST2WEIGHT_RATIO, size=4)
                self.drone.update_motor_dynamics(
                    new_motor_time_constant=mtc,
                    new_sampling_time=self.time_step,
                    new_thrust_to_weight_ratio=t2w
                )
            # set new mass and inertia to PyBullet
            self.bc.changeDynamics(
                bodyUniqueId=self.drone.body_unique_id,
                linkIndex=-1,
                mass=self.drone.m,
                localInertiaDiagonal=J_diag_sampled
            )
        else:
            pass

    @abc.abstractmethod
    def compute_done(self) -> bool:
        """Implemented by child classes."""
        raise NotImplementedError

    def compute_history(self) -> np.ndarray:
        """Returns the history of the N last observations and actions.."""

        obs_next = self.compute_observation()
        self.observation_history.append(obs_next)

        # Concatenate state-action history of size H into one column vector:
        # history = [x(k-H)^T, u(k-H-1)^T, ..., x(k), u(k-1)]^T
        history = np.concatenate([np.concatenate([o, a]) for o, a, in zip(
            self.observation_history, self.action_history)])

        # append action to action_history after history calculation such that
        # [..., x(k), u(k-1)]^T instead of  [..., x(k), u(k)]^T
        action = self.drone.last_action
        self.action_history.append(action)

        return history

    @abc.abstractmethod
    def compute_info(self) -> dict:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_observation(self) -> np.ndarray:
        """Returns the current observation of the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_potential(self) -> float:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reward(self, action) -> float:
        """Implemented by child classes."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_reference_trajectory(self):
        """Implemented by child classes."""
        raise NotImplementedError

    def render(
            self,
            mode='human'
    ) -> np.ndarray:
        """Show PyBullet GUI visualization.

        Render function triggers the PyBullet GUI visualization.
        Camera settings are managed by Task class.

        Note: For successful rendering call env.render() before env.reset()

        Parameters
        ----------
        mode: str

        Returns
        -------
        array
            holding RBG image of environment if mode == 'rgb_array'
        """
        if mode == 'human':
            # close direct connection to physics server and
            # create new instance of physics with GUI visuals
            if not self.use_graphics:
                self.bc.disconnect()
                self.use_graphics = True
                self.bc = self._setup_client_and_physics(graphics=True)
                self._setup_simulation(
                    physics=self.input_parameters['physics']
                )
                self.drone.show_local_frame()
                # Save the current PyBullet instance as save state
                # => This avoids errors when enabling rendering after training
                self.stored_state_id = self.bc.saveState()
        if mode != "rgb_array":
            return np.array([])
        else:
            raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        This function is called after agent encountered terminal state.

        Returns
        -------
        array
            holding the observation of the initial state
        """
        self.iteration = 0
        # disable rendering before resetting
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        if self.stored_state_id >= 0:
            self.bc.restoreState(self.stored_state_id)
        else:
            # Restoring a saved state circumvents the necessity to load all
            # bodies again..
            self.stored_state_id = self.bc.saveState()
        self.drone.reset()  # resets only internals such as x, y, last action
        self.task_specific_reset()
        self.apply_domain_randomization()

        # init low pass filter(s) with new values:
        self.gyro_lpf.set(x=self.drone.rpy_dot)

        # collect information from PyBullet simulation
        """Gather information from PyBullet about drone's current state."""
        self.drone.update_information()
        self.old_potential = self.compute_potential()
        self.state = self.drone.get_state()
        if self.use_graphics:  # enable rendering again after resetting
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
        obs = self.compute_observation()

        # fill history buffers
        # add observation and action to history..
        N = self.observation_history.maxlen
        [self.observation_history.append(obs) for _ in range(N)]
        action = self.drone.last_action
        [self.action_history.append(action) for _ in range(N)]
        self.last_action = action
        obs = self.compute_history()

        return obs

    def step(
            self,
            action: np.ndarray
    ) -> tuple:
        """Step the simulation's dynamics once forward.

        This method follows the interface of the OpenAI Gym.

        Parameters
        ----------
        action: array
            Holding the control commands for the agent.

        Returns
        -------
        observation (object)
            Agent's observation of the current environment
        reward (float)
            Amount of reward returned after previous action
        done (bool)
            Whether the episode has ended, handled by the time wrapper
        info (dict)
            contains auxiliary diagnostic information such as the cost signal
        """
        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            self.physics.step_forward(action)

            # Note: do not delete the following line due to >100 Hz sensor noise
            self.compute_observation()
            self.iteration += 1

        # add observation and action to history..
        next_obs = self.compute_history()

        r = self.compute_reward(action)
        info = self.compute_info()
        done = self.compute_done()
        self.last_action = action
        return next_obs, r, done, info

    @abc.abstractmethod
    def task_specific_reset(self):
        """Inheriting child classes define reset environment reset behavior."""
        raise NotImplementedError
