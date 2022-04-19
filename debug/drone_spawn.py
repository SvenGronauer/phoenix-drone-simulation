import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from pybullet_utils import bullet_client
from phoenix_drone_simulation.envs.utils import get_assets_path
from phoenix_drone_simulation.envs.agents import CrazyFlieSimpleAgent, \
    CrazyFlieBulletAgent


def main():
    SIM_FREQ = 240

    bc = bullet_client.BulletClient(connection_mode=p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,np.pi/2])

    # add open_safety_gym/envs/data to the PyBullet data path
    bc.setAdditionalSearchPath(get_assets_path())
    drone_fnp = os.path.join(get_assets_path(), 'cf21x_bullet.urdf')

    # drone_id = p.loadURDF(drone_fnp,startPos, startOrientation)
    drone = CrazyFlieBulletAgent(
        bc=bc,
        control_mode='PWM',
        time_step=1/SIM_FREQ,
        aggregate_phy_steps=1,
        motor_time_constant=0.150,
        motor_thrust_noise=0.05,
        latency=0.010,
    )
    drone_id = drone.body_unique_id
    drone.show_local_frame()

    pos = [0,0,1]

    start_pos = [0, 0, 0]
    rays_end_points = [0, 0, 0.1]
    # line_id = p.addUserDebugLine(
    #     start_pos,
    #     rays_end_points,
    #     (0.25, 0.95, 0.05),
    #     parentObjectUniqueId=drone_id,
    #     parentLinkIndex=-1,
    # )

    i = 2
    F = 0.178
    for k in range(10000):


        if k % 100 == 0:
            bc.resetBasePositionAndOrientation(
                drone.body_unique_id,
                posObj=pos,
                ornObj=startOrientation
            )
            bc.resetBaseVelocity(drone_id, angularVelocity=(-60, 0, 0))
        elif  k % 10 == 0:
            _, quat = bc.getBasePositionAndOrientation(drone_id)
            xyz_dot, rpy_dot = bc.getBaseVelocity(drone_id)
            R = np.asarray(bc.getMatrixFromQuaternion(quat)).reshape(
                (3, 3))
            rpy_dot_local = R.T @ np.array(rpy_dot)  # [rad/s]
            print(f'world rpy: {rpy_dot}')
            print(f'local rpy: {rpy_dot_local}')

            # i += 1
            # print(f'now use: i={i}')
            # xyz, abcd, _, _, _, _, xyz_dot, rpy_dot = p.getLinkState(
            #     drone_id, i, computeLinkVelocity=1)
            # print(f'Link={i} at pos: {xyz}')

        # p.applyExternalForce(
        #     drone_id,
        #     i,
        #     forceObj=[0, 0, F],
        #     posObj=[0, 0, 0],
        #     flags=p.LINK_FRAME
        # )
        # line_id = p.addUserDebugLine(
        #     start_pos,
        #     rays_end_points,
        #     (0.25, 0.95, 0.05),
        #     parentObjectUniqueId=drone_id,
        #     parentLinkIndex=-1,
        #     replaceItemUniqueId=line_id
        # )
        p.stepSimulation()
        time.sleep(1. / 15.)


if __name__ == '__main__':
    main()
