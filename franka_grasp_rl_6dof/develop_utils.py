import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import open3d as o3d
import pybullet as p
import pybullet_data
import time


def analyze_mask(mask: np.array):
    mask = mask.astype(int)
    unique_values = np.unique(mask)  # 获取 mask 中所有唯一值
    
    # 生成随机颜色，并确保不同值颜色不同
    np.random.seed(42)  # 固定随机种子以获得稳定颜色
    colors = np.random.rand(len(unique_values), 3)  # 生成 RGB 颜色
    cmap = ListedColormap(colors)  # 创建自定义 colormap
    
    # 生成 BoundaryNorm，确保颜色正确映射
    norm = BoundaryNorm(np.arange(len(unique_values) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    img = ax.imshow(mask, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')

    # 创建颜色-值映射
    color_dict = {val: colors[i] for i, val in enumerate(unique_values)}
    patches = [mpatches.Patch(color=color_dict[val], label=f'Value {val}') for val in unique_values]

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Mask Values")
    ax.set_title("Mask Visualization with Correct Colors")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()



def visualize_point_cloud(point_clouds):
    # 可视化点云

    all_pcd = {}
    for i,point_cloud in enumerate(point_clouds):
        all_pcd[str(i)] = o3d.geometry.PointCloud()
        all_pcd[str(i)].points = o3d.utility.Vector3dVector(point_cloud)
        color = [1,0,0] if i==0 else ([0,0,1] if i==1 else ([0,1,0] if i==2 else np.random.rand(3)))
        all_pcd[str(i)].paint_uniform_color(color) 
    pcds = list(all_pcd.values())

    # 世界坐标系原点
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 创建矩形顶点
    width = 0.4
    rectangle_vertices = np.array([
       [-0.5, 0],
       [-0.5, 1],
       [0.5, 1],
       [0.5, 0]
    ])
    # 创建矩形边的索引
    rectangle_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]]

    # 创建 LineSet 对象
    rectangle_line_set = o3d.geometry.LineSet()
    rectangle_line_set.points = o3d.utility.Vector3dVector(rectangle_vertices)
    rectangle_line_set.lines = o3d.utility.Vector2iVector(rectangle_lines)

    # 可视化点云和坐标系  
    o3d.visualization.draw_geometries(
        pcds + [world_frame, rectangle_line_set], 
        window_name="Point Cloud with Multiple Coordinate Frames",
        width=1280, height=960)






def calculate_joint_states_based_on_ee_placement(ee_position, ee_euler):
    # 根据末端执行器姿态返回关节角度
    def set_position(robot_id, joint_indices, joint_forces, target_position, target_orientation, ee_link):
        """控制机械臂的关节角度，使末端执行器达到目标位置和姿态"""
        joint_positions = p.calculateInverseKinematics(
            robot_id,
            ee_link,
            targetPosition=target_position,
            targetOrientation=target_orientation)
        p.setJointMotorControlArray(
            robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=joint_forces)
    
        for _ in range(480):
            p.stepSimulation()
            time.sleep(1/240)
        
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf", [0, 0, 0])
    franka_id = p.loadURDF(
        "/home/kaifeng/FYP/franka_grasp_rl_6dof/models/panda_franka/panda_modified.urdf",
        [0, 0, 0], useFixedBase=True
    )
    p.resetBasePositionAndOrientation(franka_id, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, -10)

    ee_link = 11
    neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
    joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])

    # 复位关节到中立位姿
    for joint, angle in zip(joint_indices, neutral_joint_values):
        p.resetJointState(franka_id, joint, angle)

    ee_quaternion = p.getQuaternionFromEuler(ee_euler)
    set_position(franka_id, joint_indices, joint_forces, ee_position, ee_quaternion, ee_link)

    joint_states = p.getJointStates(franka_id, joint_indices)
    joint_positions = np.array([state[0] for state in joint_states])
    y = [f'{x:.3f}' for x in joint_positions]
    print(f'\n---ee_position:{ee_position}---ee_euler:{ee_euler}---')
    print(f'---joint states:{y}---')

    p.disconnect()






if __name__ == '__main__':

    calculate_joint_states_based_on_ee_placement(ee_position=[0.4, 0, 0.4], ee_euler=[np.pi, 0, 0])