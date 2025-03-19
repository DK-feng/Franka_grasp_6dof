import pybullet as p
from gymnasium import spaces
from typing import Optional
import numpy as np
import time
import open3d as o3d
from .utils.develop_utils import *
from .utils.utils import *
import glob
from .UCN.UCN import UCN
import torch
import gymnasium as gym
import os


'''
        删除重新载入物体会有视觉残留   :已解决    
        y轴方向的运动会导致点云错位发生,不知道原因
        机械臂的初始位置是否正确,摄像头位置是否正确
        需要修改_modify_mask函数   :已解决   

'''


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def record_and_reset(self):
        time_cost = time.time() - self.start_time
        self.reset()
        return time_cost



class Panda(gym.Env):
    # robot class
    def __init__(self,
                 base_position: Optional[np.array] = (0, 0, 0),
                 image_size: Optional[np.array] = np.array([640, 480]),
                 debug_visulization: int = False):

        self.base_position = base_position if base_position is not None else (0, 0, 0)
        self.ee_link = 11
        self.camera_link = 13
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.finger_indices = np.array([9, 10])
        self.joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
        #self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.neutral_joint_values = np.array([0.000, -0.671, -0.000, -2.384, -0.000, 1.880, 0.786, -0.000, 0.000])     # 0.35, 0.4, 10度
        #self.neutral_joint_values = np.array([0.000, -0.441, 0.000, -2.180, 0.000, 1.735, 0.785, -0.000, 0.000])  # 0.4, 0.4, 0度 

        # 摄像头参数（模拟 Realsense 摄像头）
        self.camera_width = image_size[0]  # 宽度,列数
        self.camera_height = image_size[1] # 高度,行数
        self.fov = 58 
        self.aspect_ratio = self.camera_width / self.camera_height
        self.nearVal = 0.1
        self.farVal = 10
        self.debug_visulization = debug_visulization

        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect_ratio, self.nearVal, self.farVal)
        self.camera_intrinsic = self.get_camera_intrinsic()

        print("---Robot Initialization: Done---------")

    def render_from_camera(self):
        '''
        return:
            rgb_image:      np.array of shape (camera_height, camera_width, 3), 0-1
            depth_image:    np.array of shape (camera_height, camera_width)
            true_mask:      np.array of shape (camera_height, camera_width)
            mask:           np.array of shape (camera_height, camera_width)
        '''
        camera_position = self.get_camera_position()
        camera_orientation_rotate = self.get_camera_rotation_matrix()
        forward_vec = camera_orientation_rotate.dot(np.array((0, 0, -1)))
        up_vec = camera_orientation_rotate.dot(np.array((0, 1, 0)))
        target_position = camera_position + 0.1 * forward_vec

        if self.debug_visulization:
            p.addUserDebugLine(camera_position, target_position, [255,0,0], 0.2, lifeTime=0.1)

        self.view_matrix = p.computeViewMatrix(camera_position, target_position, up_vec)


        if self.rendering:
            width, height, rgb_image, depth_image, true_mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        else:
            width, height, rgb_image, depth_image, true_mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, renderer=p.ER_TINY_RENDERER)

        rgb_image = np.array(rgb_image).reshape((self.camera_height, self.camera_width, 4))[:, :, :3]
        depth_image = np.array(depth_image).reshape((self.camera_height, self.camera_width))

        # 从非线性的 OpenGL 深度图(值为0-1)中提取真实的线性深度信息（实际距离），单位为m
        # depth_image = depth_image * (self.farVal - self.nearVal) + self.nearVal
        depth_image = self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depth_image)
        # depth_image = 2.0 * self.nearVal * self.farVal / (self.farVal + self.nearVal - (2.0 * depth_image - 1.0) * (self.farVal - self.nearVal))


        rgb_image = rgb_image.astype(np.float32) / 255.0
        depth_image = depth_image.astype(np.float32)


        return rgb_image, depth_image, true_mask
    
    def reset(self):
        self._load_robot()
        self.set_joint_positions(self.neutral_joint_values)
        for _ in range(500):
            p.stepSimulation()
        self.set_joint_angles_hard(self.neutral_joint_values)
           
    def set_action(self, action: np.array, interpolation_steps=10):
        """
        在当前末端状态和目标状态之间插值。
        interpolation_steps: 插值分成多少步，每步执行一次关节控制 + p.stepSimulation()
        """
        touch_floor = False
        if len(action) == 6:
            action = np.concatenate([action, [0]])

        # 当前状态
        current_pos = np.array(p.getLinkState(self.RobotID, self.ee_link)[0])
        current_ori = np.array(p.getEulerFromQuaternion(p.getLinkState(self.RobotID, self.ee_link)[1]))

        # 目标状态
        target_pos = current_pos + action[:3] * 0.02
        # 限制最低高度
        if target_pos[2] < 0.005:
            target_pos[2] = 0.005
            touch_floor = True
        target_ori = current_ori + action[3:6] * 0.02


        # 当前手爪宽度
        current_finger = self.get_fingers_width() / 2
        target_finger = current_finger + action[-1] * 0.02

        # 插值步骤
        for step in range(1, interpolation_steps + 1):
            ratio = step / interpolation_steps

            # 插值末端位姿（线性插值）
            interp_pos = (1 - ratio) * current_pos + ratio * target_pos
            interp_ori_euler = (1 - ratio) * current_ori + ratio * target_ori
            interp_ori_quat = p.getQuaternionFromEuler(interp_ori_euler)

            # 插值手爪
            interp_finger = (1 - ratio) * current_finger + ratio * target_finger

            # 求解当前插值状态下的逆解
            joint_positions = p.calculateInverseKinematics(
                self.RobotID,
                self.ee_link,
                targetPosition=interp_pos,
                targetOrientation=interp_ori_quat,
                lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                jointRanges=[5.8] * 7,
                restPoses=self.get_joint_positions(),  # 当前状态作为启发
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            joint_positions = np.array(joint_positions)
            
            # 设置手指
            joint_positions[-2:] = interp_finger

            zero_velocities = [0.0] * len(self.joint_indices)
            p.setJointMotorControlArray(self.RobotID,
                                        jointIndices=self.joint_indices,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=joint_positions,
                                        targetVelocities=zero_velocities,
                                        forces=self.joint_forces)

            # 仿真步进
            for _ in range(self.sub_steps):
                p.stepSimulation()

        return touch_floor
                
    def _load_robot(self):
        file_path = os.path.join(os.path.dirname(__file__), "models/panda_franka/panda_modified.urdf")
        self.RobotID = p.loadURDF(file_path, self.base_position, useFixedBase=True)
        p.changeDynamics(self.RobotID, self.finger_indices[0], lateralFriction=1.0, spinningFriction=0.001)
        p.changeDynamics(self.RobotID, self.finger_indices[1], lateralFriction=1.0, spinningFriction=0.001)

    def set_joint_angles_hard(self, joint_angles):
        '''不影响仿真的动力学行为'''
        for joint,angle in zip(self.joint_indices, joint_angles):
            p.resetJointState(self.RobotID, joint, angle)

    def set_joint_positions(self, joint_positions: np.array):
        p.setJointMotorControlArray(self.RobotID,
                                    jointIndices=self.joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_positions,
                                    forces=self.joint_forces)
        
    def get_joint_positions(self):
        joint_states = p.getJointStates(self.RobotID, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states])
        return joint_positions

    def get_camera_position(self):
        return np.array(p.getLinkState(self.RobotID, self.camera_link)[0])

    def get_ee_position(self):
        return np.array(p.getLinkState(self.RobotID, self.ee_link)[0])

    def get_camera_rotation_matrix(self):
        quaternion = p.getLinkState(self.RobotID, self.camera_link)[1]
        return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)

    def get_ee_rotation_matrix(self):
        quaternion = p.getLinkState(self.RobotID, self.ee_link)[1]
        return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)

    def get_ee_euler(self):
        quaternion = p.getLinkState(self.RobotID, self.ee_link)[1]
        return np.array(p.getEulerFromQuaternion(quaternion))

    def get_camera_intrinsic(self):
        intrinsic_matrix = np.eye(3)
        # fov 为垂直视场角（单位：度）
        fov_rad = np.deg2rad(self.fov)
        f_x = self.camera_width / (2 * np.tan(fov_rad / 2))
        f_y = f_x / self.aspect_ratio
        cx = self.camera_width / 2
        cy = self.camera_height / 2

        intrinsic_matrix[0, 0] = f_x
        intrinsic_matrix[1, 1] = f_y
        intrinsic_matrix[0, 2] = cx
        intrinsic_matrix[1, 2] = cy
        return intrinsic_matrix

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = p.getJointState(self.RobotID, self.finger_indices[0])[0]
        finger2 = p.getJointState(self.RobotID, self.finger_indices[1])[0]
        return finger1 + finger2

    def get_joint_angles(self):
        '''返回每个关节的角度,np.array(9,)'''
        joint_states = [p.getJointState(self.RobotID, i) for i in self.joint_indices]
        joint_angles = np.array([state[0] for state in joint_states])
        return joint_angles




class Task(gym.Env):
    # task class
    def __init__(self,
                 num_objects: int = 4,
                 use_urdf: bool = True):
        self.num_objects = num_objects
        self.objects_IDs = []
        self.objects_urdf_file_folder = os.path.join(os.path.dirname(__file__), "models/objects")
        self.objects_files = np.array([file for file in glob.glob(os.path.join(self.objects_urdf_file_folder,"*"))
                                        if file.split('/')[-1].startswith('0')])
        self.objects_files = sorted(self.objects_files)
        self.use_urdf = use_urdf
        '''
        /home/kaifeng/FYP/data/objects/002_master_chef_can
        /home/kaifeng/FYP/data/objects/003_cracker_box
        /home/kaifeng/FYP/data/objects/004_sugar_box
        /home/kaifeng/FYP/data/objects/005_tomato_soup_can
        /home/kaifeng/FYP/data/objects/006_mustard_bottle
        /home/kaifeng/FYP/data/objects/007_tuna_fish_can
        /home/kaifeng/FYP/data/objects/008_pudding_box
        '''
        self.drop_positions = np.array([0.55, 0, 0.15])  #所有物体从这个高度自由落下形成structure cluster

        print("---Task Initialization: Done---------")

    def reset(self):
        """
        加载任务所需的物体，例如在堆积场景中加载几个 cube 模型
        """
        self.objects_IDs = self._load_objs()

    def _check_collision_and_position(self, obj_id, objects_IDs):
        not_on_table = False
        collision = False

        # 检查是否与其他物体重合
        if len(objects_IDs) == 0:
            collision = False
        else:
            for id in objects_IDs:
                if id == obj_id:
                    continue
                contact_points = p.getContactPoints(obj_id, id)
                if len(contact_points) > 0:
                    collision = True
                    break
        
        # 检查是否在台面上
        position = p.getBasePositionAndOrientation(obj_id)[0]
        if position[2] < -0.1 or position[0] < 0.25 or position[0] > 0.8 or position[1] < -0.3 or position[1] > 0.3:
            not_on_table = True

        return collision or not_on_table

    def _load_objs(self):
        objects_IDs = []
        select_objects_paths = np.random.choice(self.objects_files, size=self.num_objects, replace=False)
        if self.use_urdf:
            for file_path in select_objects_paths:
                if self.num_objects == 1:
                    position = self.drop_positions + np.array([np.random.uniform(-0.05, 0.05),
                                                    np.random.uniform(-0.05, 0.05),
                                                    0])
                else:
                    position = self.drop_positions + np.array([np.random.uniform(-0.1, 0.1),
                                                    np.random.uniform(-0.1, 0.1),
                                                    0])                    
                orientation = np.random.rand(4)
                obj_id = p.loadURDF(os.path.join(file_path,'model_normalized.urdf'),
                    basePosition = position,
                    baseOrientation = orientation)
                for _ in range(1000):
                    p.stepSimulation()
                objects_IDs.append(obj_id)

            # 一直尝试放置直到不与其他物体重合,以及排除掉下台面的情况
            DONE = False
            while not DONE:
                DONE = True
                for obj_id in objects_IDs:
                    if self._check_collision_and_position(obj_id, objects_IDs):
                        position = self.drop_positions + np.array([np.random.uniform(-0.1, 0.1),
                                                np.random.uniform(-0.1, 0.1),
                                                0])
                        orientation = np.random.rand(4)
                        p.resetBasePositionAndOrientation(obj_id, position, orientation)
                        for _ in range(10000):
                            p.stepSimulation()
                        DONE = False
                        break


            print('------Loading Object:{}---ID:{}---'.format(file_path.split('/')[-1], obj_id))

            self.select_objects_paths = np.array(select_objects_paths)
            for _ in range(5000):
                p.stepSimulation()
            return objects_IDs
        
        else:
            print("ONLY SUPPORT URDF FILE")
            time.sleep(10000)


class RobotTaskEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    # Main environment class
    def __init__(
                    self, 
                    render_mode: str='human',
                    num_objects: int=1,
                    background_color : Optional[np.array] = None,
                    timestep = 1/500,
                    sub_steps: int=20,
                    time_sleep: bool=False,
                    image_size: Optional[np.array] = np.array([640, 480]),
                    debug_visulization: bool = False,
                    pt_accumulate_ratio: float = 0.98,
                    fixed_num_points: int = 1024,
                    split: float = 0.5,
                    plane_points: int = 0,
                    points_per_frame: int = 512,
                    total_step: int = 30
                    ) -> None:

        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.background_color = background_color.astype(np.float32) / 255
        self.rendering = True if render_mode == 'human' else False
        self.render_mode = render_mode
        self.timestep = timestep
        self.sub_steps = sub_steps 
        self.time_sleep = time_sleep
        self.image_size = image_size
        self.num_objects = num_objects
        self.debug_visulization = debug_visulization
        self.split = split if num_objects > 1 else 1
        self.points_per_frame = points_per_frame
        self.plane_points = plane_points
        self.total_step = total_step

        self.return_plane_points = False if plane_points == 0 else True

        self._pt_accumulate_ratio = pt_accumulate_ratio
        self._fixed_num_points = fixed_num_points

        self.exist_obstacal = True if self.num_objects > 1 else False

        # connect to server
        options = "--background_color_red={} --background_color_green={} --background_color_blue={} --opengl3".format(
            *self.background_color
        )
        if self.rendering:
            p.connect(p.GUI, options=options)
        else:
            p.connect(p.DIRECT) 

        self.robot = Panda(image_size=self.image_size,
                            debug_visulization=self.debug_visulization)
        self.task = Task(num_objects=self.num_objects)

        self.camera_intrinsic = self.robot.get_camera_intrinsic()

        # Segementation Network
        self.UCN = UCN(self.camera_intrinsic)


        # Obs
        observation, _ = self.reset()

        all_pc_shape = observation["all_PC"].shape
        all_pc_dtype = observation["all_PC"].dtype
        joint_state_shape = observation["joint_state"].shape
        joint_state_dtype = observation["joint_state"].dtype
        ee_state_shape = observation["ee_state"].shape
        ee_state_dtype = observation["ee_state"].dtype
        
        self.observation_space = spaces.Dict(
            dict(
                all_PC=spaces.Box(-1.0, 1.0, shape=all_pc_shape, dtype=all_pc_dtype),
                joint_state=spaces.Box(-4.0, 4.0, shape=joint_state_shape, dtype=joint_state_dtype), 
                ee_state=spaces.Box(-4.0, 4.0, shape=ee_state_shape, dtype=ee_state_dtype),
                timestep=spaces.Discrete(total_step)))

        # 6D action space: [dx, dy, dz, d_roll, d_pitch, d_yaw]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -np.pi, -np.pi, -np.pi]),
            high=np.array([1, 1, 1, np.pi, np.pi, np.pi]),
            dtype=np.float32)


        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self._place_visualizer()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        print("---RobotTaskEnv Initialization: Done---------")

    def _get_obs(self):
        # 捕获初始摄像头点云并累计 0.02S
        
        rgb_image, depth_image, true_mask = self.robot.render_from_camera()
        
        # 从UCN得到mask，0.085S
        with torch.no_grad():
            mask, _ = self.UCN.get_mask(rgb_image,
                                        depth_image,
                                        num_objects=self.num_objects + 1)  #别忘了加上桌面
        mask = mask.cpu().numpy().squeeze()
        #analyze_mask(mask)
        
        # 0.03S
        # 第一次时采用mask中面积最大的value除去背景,目标物体设置为0,所有障碍物设置为1,桌面和背景设置为50
        if self._env_step == 0:
            target_mask_value = get_target_obj_from_mask(mask)
            mask = process_mask(mask, target_mask_value)
        else:
            mask = self._modify_mask(mask, depth_image)
        self.mask = mask
        
        # analyze_mask(mask)
        # 得到选择的抓取物体的正确ID,几乎0S
        if self._env_step == 0:
            self.target_object_ID = self._get_target_object_ID(mask, true_mask)

        # [N, 3], 从depth_image里生成基于camera frame的目标物体的point_cloud, 0.02S
        target_points, obstacal_points, plane_points = self._generate_pc_from_depth(depth_image, mask)
        
        # [N, 3], 将pc转换到world frame下，几乎0S
        
        target_points_worldframe = self._camera_to_world(target_points)
        obstacal_points_worldframe = self._camera_to_world(obstacal_points)
        plane_points_worldframe = self._camera_to_world(plane_points)
        

        # [N + new points, 3], 先转换为世界坐标系下再累积，0S
        self.update_curr_acc_target_points(target_points_worldframe)
        if self.exist_obstacal:
            self.update_curr_acc_obstacal_points(obstacal_points_worldframe)
        if self.return_plane_points:
            self.update_curr_acc_plane_points(plane_points_worldframe)
        

        # 向上/下采样为固定点数，0.2S,优化后0.06S
        self.curr_acc_target_points = regularize_pc_point_count(self.curr_acc_target_points, self._fixed_num_points * self.split, use_farthest_point=True)

        if self.exist_obstacal:
            self.curr_acc_obstacal_points = regularize_pc_point_count(self.curr_acc_obstacal_points, self._fixed_num_points * (1-self.split) - self.plane_points, use_farthest_point=True)
        if self.return_plane_points:
            self.curr_acc_plane_points = regularize_pc_point_count(self.curr_acc_plane_points, self.plane_points, use_farthest_point=True)

        # 所有的点云,将target的点云上移5cm,其它点云下降5cm以区分,同时归一化处理，0S
        all_PC = normalize_point_cloud(np.concatenate([self.curr_acc_target_points+np.array([0,0,0.05]),
                                        self.curr_acc_obstacal_points+np.array([0,0,-0.05]),
                                        self.curr_acc_plane_points+np.array([0,0,-0.05])], axis=0))

        ee_state = np.concatenate((self.robot.get_ee_position(), self.robot.get_ee_euler()))

        obs = { 'all_PC': all_PC, #处理后的所有点云合并在一起
                #'target_PC': self.curr_acc_target_points,  # (fixed_num_points * split, 3)
                #'obstacal_PC': self.curr_acc_obstacal_points,  # (fixed_num_points * (1-split) - plane_points, 3)
                #'plane_PC': self.curr_acc_plane_points,  # (plane_points, 3)
                'timestep': self._env_step,  # 0~29
                'joint_state': self.robot.get_joint_positions()[:7],    # (7,)
                'ee_state': ee_state}  # (x, y, z, rx, ry, rz)

        return obs

    def reset(self, seed=None, options=None):
        """
        重置仿真：重置 PyBullet 环境、重新加载平面、机械臂和任务场景，
        并清空累计点云数据，返回初始 observation。
        """
        timer = Timer()
        p.resetSimulation()
        self._reset_pybullet()
        print('\n---Reset Pybullet: Done--- time cost:{:.3f}---'.format(timer.record_and_reset()))
        self._create_scene()
        print('---Create Scene: Done--- time cost:{:.3f}---'.format(timer.record_and_reset()))
        self.robot.reset()
        print('---Load Robot: Done--- time cost:{:.3f}---'.format(timer.record_and_reset()))
        self.task.reset()
        print('---Load Objects: Done--- time cost:{:.3f}---'.format(timer.record_and_reset()))

        # 清空共享文件夹
        files = glob.glob(os.path.join('/home/kaifeng/FYP/docker_shared',"*"))
        for file in files:
            os.remove(file)

        # task 与 robot 通讯
        self.robot.sub_steps = self.sub_steps
        self.robot.rendering = self.rendering

        # reset后清空累积点云
        self.curr_acc_target_points = np.zeros([0, 3])
        self.curr_acc_obstacal_points = np.zeros([0, 3])
        self.curr_acc_plane_points = np.zeros([0, 3])
        self._env_step = 0
        
        obs = self._get_obs()

        info = {'env_step': self._env_step}

        return obs, info

    def step(self, action):
        """
        执行一个动作，返回 observation, reward, done, info
        """
        # 0.02S
        touch_floor = self.robot.set_action(action, interpolation_steps=20)
        for _ in range(self.sub_steps):
            p.stepSimulation()
            if self.time_sleep:
                time.sleep(self.timestep)

        self._env_step += 1

        truncated = bool(self._env_step >= self.total_step)

        # 0.24S / 0.275S
        obs = self._get_obs()

        # 如果末端执行器与点云中最近的点距离小于10cm则terminated
        terminated, try_grasp = self._check_terminated()
        terminated = terminated or touch_floor

        if try_grasp:
            # 接近
            grasp_action1 = 0.115 * self.robot.get_camera_rotation_matrix().dot(np.array((0, 0, -1)))
            grasp_action1 = np.concatenate([grasp_action1*50, np.array([0, 0, 0, 0.04*50])])
            self.robot.set_action(grasp_action1, 100)
            for _ in range(self.sub_steps):
                p.stepSimulation()
                time.sleep(1/500)
            # 抓取
            grasp_action2 = np.array([0, 0, 0, 0, 0, 0, -0.04*50])
            self.robot.set_action(grasp_action2, 10)
            for _ in range(self.sub_steps):
                p.stepSimulation()
                time.sleep(1/500)
            # 提升
            grasp_action3 = np.array([0, 0, 0.2*50, 0, 0, 0, -0.1])
            self.robot.set_action(grasp_action3, 100)
            for _ in range(self.sub_steps):
                p.stepSimulation()
                time.sleep(1/500)

        reward = self.compute_reward()
        info = {'env_step': self._env_step}

        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        # 目标物体高度大于20cm则成功
        reward = self._get_grasp_reward() + 2*self._get_distance_reward() + self._get_track_reward()
        return reward

    def close(self):
        p.disconnect()



    def _check_terminated(self):
        ee_position = self.robot.get_ee_position()
        terminated = True if ee_position[0] < 0.2 or ee_position[1] < -0.5 or ee_position[1] > 0.5 or ee_position[2] > 0.8 else False

        l2_distance = np.linalg.norm(ee_position - self.curr_acc_target_points, axis=1)
        try_grasp =  True if l2_distance.min() < 0.1 else False
        terminated = True if try_grasp else False
        return terminated, try_grasp

    def _get_grasp_reward(self):
        # 抓取成功reward为5,失败为-1
        height = self._get_target_obj_position()[2]
        reward = 2.0 if height > 0.1 else -1.0
        return reward

    def _get_distance_reward(self):
        # 小于0.1reward为0,大于时为距离的负数
        ee_position = self.robot.get_ee_position()
        target_position = self._get_target_obj_position()
        l2_distance = np.linalg.norm(ee_position - self.curr_acc_target_points, axis=1)
        center_dis = np.linalg.norm(ee_position - target_position)
        reward = 0.0 if l2_distance.min() < 0.1 else -(0.5*center_dis + 0.5*l2_distance.min())
        return reward

    def _get_track_reward(self):
        # 处理后的mask需要包含至少50个目标物体的像素点
        target_value_count = np.count_nonzero(self.mask == 0)
        reward = -1.0 if target_value_count < 50 else 0.0
        return reward

    def save_state(self):
        state_id = p.saveState()
        return state_id
    
    def restore_state(self, state_id):
        p.restoreState(state_id)

    def remove_state(self, state_id):
        p.removeState(state_id)

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.
        If render mode is "rgb_array" or "human, return an RGB array of the scene. Else, do nothing and return None.
        Args:
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.
            mode (str, optional): Deprecated: This argument is deprecated and will be removed in a future
                version. Use the render_mode argument of the constructor instead.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if self.render_mode == "rgb_array" or 'human':
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            # With Python3.10, pybullet return flat tuple instead of array. So we need to build create the array.
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]

    def _place_visualizer(self,
                            render_distance: float = 1.4,
                            render_yaw: float = 45,
                            render_pitch: float = -30,
                            render_target_position: Optional[np.ndarray] = None):
        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=render_distance,
                cameraYaw=render_yaw,
                cameraPitch=render_pitch,
                cameraTargetPosition=render_target_position if render_target_position is not None else [0, 0, 0]
            )

    def _get_target_obj_position(self):
        return p.getBasePositionAndOrientation(self.target_object_ID)[0]

    def _create_scene(self):
        self.PlaneID = p.loadURDF("/home/kaifeng/FYP/franka_grasp_rl_6dof/models/sense_file/floor/model_normalized.urdf", basePosition=[0,0,-0.5])
        self.TableID = create_table(table_length=1.8,
                                    table_width=1,
                                    table_height=0.5,
                                    table_position=[0.5,0,-0.5])

    def _reset_pybullet(self):
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timestep)

    def _get_target_object_ID(self, mask, true_mask):
        '''比较UCN的mask和Pybullet真实的mask来得到目标物体的pybullet环境中的ID'''
        conjunction_part = true_mask[mask==0]
        unique_values, counts = np.unique(conjunction_part, return_counts=True)
        if len(counts) < 2:
            return unique_values[0]
        target_object_ID = unique_values[np.argmax(counts)]
        return target_object_ID

    def _modify_mask(self, mask, depth_image, threshold_distance=0.1, min_match_points=10, num_threshold=30):
        '''让前后生成的mask中对目标物体的value值不变,每个点的threshold_distance内有min_match_points个点与累积target_PC中的点的距离小于阈值'''
        mask = mask.copy()
        target_mask_value = None
        unique_values = np.sort(np.unique(mask))[1:]
        num_points_count = []
        for value in unique_values:
            _mask = process_mask(mask, value)
            _point_clouds = self._generate_pc_from_depth(depth_image, _mask, return_obstacal_and_plane=False)
            _point_clouds_worldframe = self._camera_to_world(_point_clouds)
            if _point_clouds_worldframe.shape[0] < 400:
                continue

            # 随机在当前value生成的点云中选取200个点
            index = np.random.choice(range(_point_clouds_worldframe.shape[0]), size=400, replace=False)
            _pc_sample = _point_clouds_worldframe[index, :]
            _pc_sample = farthest_point_sampling_gpu(_pc_sample, 200)
            # 计算选取的200个点与累积点云中的点的L2距离
            diff = _pc_sample[:, np.newaxis, :] - self.curr_acc_target_points[np.newaxis, :, :] 
            distance_matrix = np.linalg.norm(diff, axis=-1)

            # 计算100个点中有多少个点与累积点云中的点的距离小于阈值
            num_valid_points = np.sum(np.sum(distance_matrix < threshold_distance, axis=1) > min_match_points)
            num_points_count.append(num_valid_points)
            # 如果至少有min_match_points个点与累积点云中的点的距离小于阈值,说明这个value是目标物体的value

        # 100个点中有一半以上的点在0.03范围内有至少10个累积点云,否则不是目标物体
        if len(num_points_count) != 0:
            target_mask_value = unique_values[np.argmax(num_points_count)] if max(num_points_count) > num_threshold else 666
        else:
            target_mask_value = 666
        mask = process_mask(mask, target_mask_value)

        return mask

    def _generate_pc_from_depth(self, depth_image, target_mask, return_obstacal_and_plane: bool=True):
        # 返回camera frame的 target_pc 和 obstacle_pc 和 plane_pc

        if target_mask is None:
            return None

        # 从深度图像生成camera_frame的PC
        fx, fy = self.camera_intrinsic[0, 0], self.camera_intrinsic[1, 1]
        cx, cy = self.camera_intrinsic[0, 2], self.camera_intrinsic[1, 2]

        xs, ys = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]))
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)
        x_normalized = (xs - cx) / fx
        y_normalized = (ys - cy) / fy

        # all_points: (480, 640, 3)
        all_points = np.stack((x_normalized * np.squeeze(depth_image),
                                y_normalized * -np.squeeze(depth_image),
                                -np.squeeze(depth_image)), axis=-1)
        target_points = all_points[target_mask==0]
        target_points = target_points.reshape(-1, 3)

        # 返回target,obstacl,plane的点云
        if return_obstacal_and_plane:

            obstacal_points = all_points[target_mask==1]
            obstacal_points = obstacal_points.reshape(-1, 3)
            if obstacal_points.shape[0] != 0:
                obstacal_points_indexes = np.random.choice(range(obstacal_points.shape[0]), size=self.points_per_frame, replace=(obstacal_points.shape[0] < self.points_per_frame))
                obstacal_points = obstacal_points[obstacal_points_indexes,:]

            plane_points = all_points[target_mask==50]
            if target_points.shape[0] != 0:
                target_points_indexes = np.random.choice(range(target_points.shape[0]), size=self.points_per_frame, replace=(target_points.shape[0] < self.points_per_frame))
                target_points = target_points[target_points_indexes,:]

            plane_points = plane_points.reshape(-1, 3)
            if plane_points.shape[0] != 0:
                plane_points_indexes = np.random.choice(range(plane_points.shape[0]), size=self.points_per_frame//2, replace=(plane_points.shape[0] < self.points_per_frame//2))
                plane_points = plane_points[plane_points_indexes,:]

            return target_points, obstacal_points, plane_points

        # 只返回target的点云
        else:
            if target_points.shape[0] != 0:
                points_indexes = np.random.choice(range(target_points.shape[0]), size=2 * self.points_per_frame, replace=(target_points.shape[0] < 2 * self.points_per_frame))
                target_points = target_points[points_indexes,:]
            return target_points

    def _camera_to_world(self, point_clouds):
        '''将PC从camera frame转换为world frame'''
        if point_clouds is None:
            return None
        if point_clouds.shape[0] == 0:
            return None

        # [4, N],最后一行为1
        camera_points_homogeneous = np.hstack((point_clouds, np.ones((point_clouds.shape[0], 1)))).T
        # view_matrix的逆矩阵就为外参
        view_matrix = np.array(self.robot.view_matrix).reshape(4, 4)
        camera_to_world = np.linalg.inv(view_matrix).T
        # introduction to robotics 19页
        world_points_homogeneous = camera_to_world @ camera_points_homogeneous
        # 注意形状变回(N, 3)
        world_points = world_points_homogeneous.T[:, :3]  # 只取前三个维度
        return world_points

    def update_curr_acc_target_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        # accumulate points
        if new_points is None:
            return
        if new_points.shape[0] == 0:    
            return
        new_points = bound_points(new_points, centroid=self.task.drop_positions, width=0.4)  # 限定障碍物点云的范围为中心为drop_position的边长为0.6的矩形
        index = np.random.choice(
            range(new_points.shape[0]),
            size=int(self._pt_accumulate_ratio**self._env_step * new_points.shape[0]),
            replace=False,
        ).astype(np.int32)
        self.curr_acc_target_points = np.concatenate(
            (new_points[index, :], self.curr_acc_target_points), axis=0)  

    def update_curr_acc_obstacal_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        # accumulate points
        if new_points is None:
            return
        if new_points.shape[0] == 0:    
            return
        new_points = bound_points(new_points, centroid=self.task.drop_positions, width=0.4)  # 限定障碍物点云的范围为中心为(0.6, 0, 0)的边长为0.6的矩形
        index = np.random.choice(
            range(new_points.shape[0]),
            size=int(self._pt_accumulate_ratio**self._env_step * new_points.shape[0]),
            replace=False,
        ).astype(np.int32)
        self.curr_acc_obstacal_points = np.concatenate(
            (new_points[index, :], self.curr_acc_obstacal_points), axis=0)  

    def update_curr_acc_plane_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        # accumulate points
        if new_points is None:
            return
        if new_points.shape[0] == 0:    
            return
        new_points = bound_points(new_points, centroid=self.task.drop_positions, width=0.3)  # 限定障碍物点云的范围为中心为(0.6, 0, 0)的边长为0.6的矩形
        index = np.random.choice(
            range(new_points.shape[0]),
            size=int(self._pt_accumulate_ratio**self._env_step * new_points.shape[0]),
            replace=False,
        ).astype(np.int32)
        self.curr_acc_plane_points = np.concatenate(
            (new_points[index, :], self.curr_acc_plane_points), axis=0)  

    def visualize_point_cloud(self, point_clouds=None):
        # 可视化点云
        if point_clouds is None:
            point_clouds = [self.curr_acc_target_points,self.curr_acc_obstacal_points,self.curr_acc_plane_points]

        if isinstance(point_clouds, np.ndarray):
            point_clouds = [point_clouds]

        all_pcd = {}
        for i,point_cloud in enumerate(point_clouds):
            all_pcd[str(i)] = o3d.geometry.PointCloud()
            all_pcd[str(i)].points = o3d.utility.Vector3dVector(point_cloud)
            color = [1,0,0] if i==0 else ([0,0,1] if i==1 else ([0,1,0] if i==2 else np.random.rand(3)))
            all_pcd[str(i)].paint_uniform_color(color) 
        pcds = list(all_pcd.values())

        # 世界坐标系原点
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # 相机坐标系原点
        camera_position = self.robot.get_camera_position()
        camera_rotation_matrix = self.robot.get_camera_rotation_matrix()
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=camera_position)
        camera_frame.rotate(camera_rotation_matrix, center=camera_position)

        # 机械臂末端坐标系原点
        ee_position = self.robot.get_ee_position()
        ee_rotation_matrix = self.robot.get_ee_rotation_matrix()
        ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=ee_position)
        ee_frame.rotate(ee_rotation_matrix, center=ee_position)

        # 创建矩形顶点
        width = 0.4
        rectangle_vertices = np.array([
            [self.task.drop_positions[0]-width, self.task.drop_positions[1]-width, 0],
            [self.task.drop_positions[0]-width, self.task.drop_positions[1]+width, 0],
            [self.task.drop_positions[0]+width, self.task.drop_positions[1]+width, 0],
            [self.task.drop_positions[0]+width, self.task.drop_positions[1]-width, 0]
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
            pcds + [world_frame, camera_frame, rectangle_line_set], 
            window_name="Point Cloud with Multiple Coordinate Frames",
            width=1280, height=960)

    def close(self):
        p.disconnect()



if __name__ == "__main__":

    print(os.path.join(os.path.dirname(__file__), "models/panda_franka/panda_modified.urdf"))











