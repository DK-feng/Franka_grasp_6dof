import pybullet as p
import pybullet_data
import time
import numpy as np


def set_position(robot_id,joint_indices,joint_forces, target_position, target_orientation):
    joint_positions = p.calculateInverseKinematics(robot_id,
                                                    11,
                                                    targetPosition=target_position,
                                                    targetOrientation=target_orientation)
    p.setJointMotorControlArray(robot_id,
                                jointIndices=joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_positions,
                                forces=joint_forces)
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1/240)



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf", [0, 0, 0])
frankaId = p.loadURDF("/home/kaifeng/FYP/URDF_files/panda_franka/panda_modified.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(frankaId, [0, 0, 0], [0, 0, 0, 1])

p.setGravity(0,0,-10)

ee_link = 11
neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])

for joint,angle in zip(joint_indices, neutral_joint_values):
    p.resetJointState(frankaId, joint, angle)


for i in range(100000):
    # if i == 10:
    #     print(p.getLinkState(frankaId,11))
    #     print(p.getEulerFromQuaternion(p.getLinkState(frankaId,11)[1]))

    if i == 20:
        set_position(frankaId,joint_indices,joint_forces,
                     target_position=[0.3, 0, 0.6],
                     target_orientation=p.getQuaternionFromEuler([np.pi,-np.pi/9,0]))
    p.stepSimulation()
    time.sleep(1/240)

    if i == 100:
        joint_states = p.getJointStates(frankaId, joint_indices)
        joint_positions = np.array([state[0] for state in joint_states])
        y = [f'{x:.3f}' for x in joint_positions]
        print(y)

p.disconnect()