import pybullet as p
from gymnasium import spaces
from typing import Optional
import numpy as np
import time
import open3d as o3d
from franka_grasp_rl_6dof.develop_utils import *
from franka_grasp_rl_6dof.utils import *
import glob
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
from franka_grasp_rl_6dof.UCN.UCN import UCN
import cv2
import torch
import gymnasium as gym



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
        #self.neutral_joint_values = np.array([0.000, -0.671, -0.000, -2.384, -0.000, 1.880, 0.786, -0.000, 0.000])     # 0.35, 0.4, 10度
        self.neutral_joint_values = np.array([0.000, -0.441, 0.000, -2.180, 0.000, 1.735, 0.785, -0.000, 0.000])  # 0.4, 0.4, 0度 

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
            depth_image:    np.array of shape (camera_height, camera_width, 1)
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

        # rgb_image: [height, width, 4]
        # depth_image: [height, width]
        # mask: [height, width]
        width, height, rgb_image, depth_image, true_mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_image = np.array(rgb_image).reshape((self.camera_height, self.camera_width, 4))[:, :, :3]
        depth_image = np.array(depth_image).reshape((self.camera_height, self.camera_width))

        # 从非线性的 OpenGL 深度图(值为0-1)中提取真实的线性深度信息（实际距离），单位为m
        # depth_image = depth_image * (self.farVal - self.nearVal) + self.nearVal
        depth_image = self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depth_image)
        # depth_image = 2.0 * self.nearVal * self.farVal / (self.farVal + self.nearVal - (2.0 * depth_image - 1.0) * (self.farVal - self.nearVal))

        # depth_image = depth_image[..., np.newaxis] #(480行,640列)即640宽*480高
        # mask = np.array(mask).reshape((self.camera_height, self.camera_width))
        # mask = self._process_mask(mask, self.target_object_indice)

        rgb_image = rgb_image.astype(np.float32) / 255.0
        depth_image = depth_image.astype(np.float32)

        # analyze_mask(mask) 
        # time.sleep(1000000)
        # mask = mask.cpu().numpy().squeeze().astype(np.int16)
        # analyze_mask(mask) 
        # time.sleep(1000000)

        return rgb_image, depth_image, true_mask
    
    def reset(self):
        '''
        resetJointState 只改变状态，不控制电机
        p.resetJointState 是瞬时地改变关节状态，但它不会同步更新电机控制模式中的目标位置。
        如果你在仿真步进时仍然用 POSITION_CONTROL 控制，它会尝试回到之前的目标状态。
        POSITION_CONTROL 会覆盖 resetJointState 的效果
        如果 p.setJointMotorControlArray 在 reset() 后继续运行，电机会保持之前的目标位置。
        '''
        self._load_robot()
        self.set_joint_positions(self.neutral_joint_values)
        for _ in range(5000):
            p.stepSimulation()
        self.set_joint_angles_hard(self.neutral_joint_values)
           
    def set_action(self, action: np.array):
        # action: postion change of end effector, (dx, dy, dz)
        target_position = p.getLinkState(self.RobotID, self.ee_link)[0] + action[:3]*0.05
        if target_position[2] < 0.02:
            target_position[2] = 0.02
        target_orientation = p.getEulerFromQuaternion(p.getLinkState(self.RobotID, self.ee_link)[1]) + action[3:6]*0.05
        target_orientation = p.getQuaternionFromEuler(target_orientation)
        joint_positions = p.calculateInverseKinematics(self.RobotID,
                                                        self.ee_link,
                                                        targetPosition=target_position,
                                                        targetOrientation=target_orientation)
        finger_width = self.get_fingers_width()
        finger_position = finger_width/2 + action[-1]*0.05
        joint_positions = np.array(joint_positions)
        joint_positions[-2:] = finger_position
        self.set_joint_positions(joint_positions)

    def _load_robot(self):
        self.RobotID = p.loadURDF("/home/kaifeng/FYP/franka_grasp_rl_6dof/models/panda_franka/panda_modified.urdf", self.base_position, useFixedBase=True)
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
        self.objects_urdf_file_folder = '/home/kaifeng/FYP/franka_grasp_rl_6dof/models/objects'
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
        self.drop_positions = np.array([0.5, 0, 0.15])  #所有物体从这个高度自由落下形成structure cluster

        print("---Task Initialization: Done---------")

    def is_success(self):
        """
        判断任务是否成功
        """
        return False

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
        # 随即导入新物体,需要改正，参数仅适用于米老鼠
        objects_IDs = []
        select_objects_paths = np.random.choice(self.objects_files, size=self.num_objects, replace=False)
        if self.use_urdf:
            for file_path in select_objects_paths:
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
            for obj_file in select_objects_paths:
                obj_id = self._load_obj(os.path.join(file_path,'model_normalized.obj'),
                                        self.drop_positions,
                                        np.random.rand(4), mesh_scale=[1,1,1])
                print('------Loading Object:{}---ID:{}---'.format(obj_file.split('/')[-1], obj_id))
                for _ in range(1000):
                    p.stepSimulation()
                objects_IDs.append(obj_id)
            self.select_objects_paths = np.array(select_objects_paths)
            for _ in range(5000):
                p.stepSimulation()
            return objects_IDs

    def get_objects_info(self, target_object_id):
        '''返回所加载物体的path以及pose,为制作.mat文件调用'''
        poses = []
        sorted_path = sorted(self.select_objects_paths)
        for path in sorted_path:
            obj_id = self.objects_IDs[list(self.select_objects_paths).index(path)]
            info = p.getBasePositionAndOrientation(obj_id)
            position = np.array(info[0])
            rotation_matrix = np.array(p.getMatrixFromQuaternion(info[1])).reshape(3,3)
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = position
            poses.append(pose)
        poses = np.array(poses)
        target_object_path = self.select_objects_paths[self.objects_IDs.index(target_object_id)]
        select_objects_index = np.array(sorted([self.objects_files.index(path) for path in self.select_objects_paths]))
        path = sorted(["/".join(p.split("/")[-3:]) for p in self.select_objects_paths])
        target_path = np.array(target_object_path.split("/")[-1])

        return path, poses, target_path, select_objects_index

    def _load_obj(self, obj_path, obj_position, obj_orientation, mesh_scale):
        '''导入.obj文件'''
        visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        visualFramePosition=[0,0,0],
        meshScale=mesh_scale)

        collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        collisionFramePosition=[0,0,0],
        meshScale=mesh_scale)

        objectId = p.createMultiBody(
            baseMass=1, #质量(kg),0表示固定物体
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=obj_position,
            baseOrientation=obj_orientation, #四元数
            useMaximalCoordinates=True)
        
        return objectId




class RobotTaskEnv(gym.Env):
    # Main environment class
    def __init__(
                    self, 
                    render_mode: str='human',
                    num_objects: int=3,
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
                    points_per_frame: int = 512
                    ) -> None:

        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.background_color = background_color.astype(np.float32) / 255
        self.render = True if render_mode == 'human' else False
        self.timestep = timestep
        self.sub_steps = sub_steps 
        self.time_sleep = time_sleep
        self.image_size = image_size
        self.num_objects = num_objects
        self.debug_visulization = debug_visulization
        self.split = split if num_objects > 1 else 1
        self.points_per_frame = points_per_frame
        self.plane_points = plane_points

        self.return_plane_points = False if plane_points == 0 else True

        self._pt_accumulate_ratio = pt_accumulate_ratio
        self._fixed_num_points = fixed_num_points

        self.exist_obstacal = True if self.num_objects > 2 else False

        # connect to server
        options = "--background_color_red={} --background_color_green={} --background_color_blue={} --opengl3".format(
            *self.background_color
        )
        if self.render:
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
                joint_state=spaces.Box(-2.0, 2.0, shape=joint_state_shape, dtype=joint_state_dtype), 
                ee_state=spaces.Box(-2.0, 2.0, shape=ee_state_shape, dtype=ee_state_dtype),
                timestep=spaces.Discrete(50)))

        # 6D action space: [dx, dy, dz, d_roll, d_pitch, d_yaw, claw_action]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -np.pi/3, -np.pi/3, -np.pi/3, -0.1]),
            high=np.array([1, 1, 1, np.pi/3, np.pi/3, np.pi/3, 0.1]),
            dtype=np.float32)


        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self._place_visualizer()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        print("---RobotTaskEnv Initialization: Done---------")

    def _get_obs(self):
        # 捕获初始摄像头点云并累计 0.03S
        rgb_image, depth_image, true_mask = self.robot.render_from_camera()

        # 从UCN得到mask, 0.14S
        with torch.no_grad():
            mask, _ = self.UCN.get_mask(rgb_image,
                                        depth_image,
                                        num_objects=self.num_objects + 1)  #别忘了加上桌面
        mask = mask.cpu().numpy().squeeze()
        #analyze_mask(mask)

        # 第一次时采用mask中面积最大的value除去背景,目标物体设置为0,所有障碍物设置为1,桌面和背景设置为50, 0.1S
        if self._env_step == 0:
            target_mask_value = get_target_obj_from_mask(mask)
            mask = process_mask(mask, target_mask_value)

        else:
            mask = self._modify_mask(mask, depth_image)

        # analyze_mask(mask)
        # 得到选择的抓取物体的正确ID,几乎0S
        if self._env_step == 0:
            self.target_object_ID = self._get_target_object_ID(mask, true_mask)

        # [N, 3], 从depth_image里生成基于camera frame的目标物体的point_cloud, 0.03S
        target_points, obstacal_points, plane_points = self._generate_pc_from_depth(depth_image, mask)

        print(target_points.shape)
        # [N, 3], 将pc转换到world frame下
        target_points_worldframe = self._camera_to_world(target_points)
        obstacal_points_worldframe = self._camera_to_world(obstacal_points)
        plane_points_worldframe = self._camera_to_world(plane_points)
        print(target_points_worldframe.shape)

        print(self.curr_acc_target_points.shape)
        # [N + new points, 3], 先转换为世界坐标系下再累积
        self.update_curr_acc_target_points(target_points_worldframe)
        if self.exist_obstacal:
            self.update_curr_acc_obstacal_points(obstacal_points_worldframe)
        if self.return_plane_points:
            self.update_curr_acc_plane_points(plane_points_worldframe)

        print(self.curr_acc_target_points.shape)
        timer = Timer()
        # 向上/下采样为固定点数，0.2S
        self.curr_acc_target_points = regularize_pc_point_count(self.curr_acc_target_points, self._fixed_num_points * self.split, use_farthest_point=True)
        print(timer.record_and_reset())
        if self.exist_obstacal:
            self.curr_acc_obstacal_points = regularize_pc_point_count(self.curr_acc_obstacal_points, self._fixed_num_points * (1-self.split) - self.plane_points, use_farthest_point=True)
        if self.return_plane_points:
            self.curr_acc_plane_points = regularize_pc_point_count(self.curr_acc_plane_points, self.plane_points, use_farthest_point=True)
        print(self.curr_acc_target_points.shape)

        # 所有的点云,将target的点云上移5cm,其它点云下降5cm以区分,同时归一化处理
        all_PC = normalize_point_cloud(np.concatenate([self.curr_acc_target_points+np.array([0,0,0.05]),
                                        self.curr_acc_obstacal_points+np.array([0,0,-0.05]),
                                        self.curr_acc_plane_points+np.array([0,0,-0.05])], axis=0))

        ee_state = np.concatenate((self.robot.get_ee_position(), self.robot.get_ee_euler(), [self.robot.get_fingers_width()/2]))

        obs = { 'all_PC': all_PC, #处理后的所有点云合并在一起
                #'target_PC': self.curr_acc_target_points,  # (fixed_num_points * split, 3)
                #'obstacal_PC': self.curr_acc_obstacal_points,  # (fixed_num_points * (1-split) - plane_points, 3)
                #'plane_PC': self.curr_acc_plane_points,  # (plane_points, 3)
                'timestep': self._env_step,  # 1~50
                'joint_state': self.robot.get_joint_positions()[:7],    # (7,)
                'ee_state': ee_state}  # (x, y, z, rx, ry, rz, finger_width/2)

        return obs

    def reset(self):
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
        self.robot.set_action(action)
        for _ in range(self.sub_steps):
            p.stepSimulation()
            if self.time_sleep:
                time.sleep(self.timestep)

        self._env_step += 1

        truncated = bool(self._env_step > 50)
        terminated = bool(self._get_target_obj_height() > 0.2)  # 目标物体高度大于20cm则成功

        # 0.4S
        obs = self._get_obs()


        reward = self.compute_reward()
        info = {'env_step': self._env_step}

        return obs, reward, terminated, truncated, info

    def get_expert_actions(self):
        self._generate_mat_file()
        position_array = self._contact_expert()
        return position_array

    def compute_reward(self):
        # 目标物体高度大于20cm则成功
        height = self._get_target_obj_height()
        return -np.array(height > 0.2, dtype=np.float32)

    def close(self):
        p.disconnect()


    def save_state(self):
        state_id = p.saveState()
        return state_id
    
    def restore_state(self, state_id):
        p.restoreState(state_id)

    def remove_state(self, state_id):
        p.removeState(state_id)

    def close(self):
        p.disconnect()

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

    def _get_target_obj_height(self):
        return p.getBasePositionAndOrientation(self.target_object_ID)[0][2]

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

    def step_expert(self, joint_positions):
        """
        执行一个动作，返回 observation, reward, done, info
        """
        current_state = np.concatenate((self.robot.get_ee_position(), self.robot.get_ee_euler(), [self.robot.get_fingers_width()/2]))
        
        self.robot.set_joint_positions(joint_positions)
        for _ in range(self.sub_steps):
            p.stepSimulation()
            if self.time_sleep:
                time.sleep(self.timestep)

        target_states = np.concatenate((self.robot.get_ee_position(), self.robot.get_ee_euler(), [self.robot.get_fingers_width()/2]))
        expert_action = target_states - current_state

        self._env_step += 1
        obs = self._get_obs()
        reward = self.compute_reward()

        truncated = bool(self._env_step > 50)
        terminated = bool(self._get_target_obj_height() > 0.2)

        return obs, reward, terminated, truncated, {}, expert_action

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

    def _generate_mat_file(self):    
        '''
        path: (num,), 只写
        pose: (num, 4, 4), trans = pose[:3, 3], orn = pose[:3, :3]
        states: (joint_states, 0, ee_states) -> shape(1,10)
        target_name: 是path里的一部分
        used_objects_index: 所是用的物体的path在sorted_all_paths里的index
        '''
        path, pose, target_path, used_objects_index = self.task.get_objects_info(target_object_id=self.target_object_ID)
        states = np.insert(self.robot.neutral_joint_values, 7, 0)
        mat_data = {
            "path": path,
            "pose": pose,
            "states": states,
            "target_name": target_path,
            "used_objects_index": used_objects_index
        }
        savemat("/home/kaifeng/FYP/docker_shared/sence_data.mat", mat_data)
        # savemat("/home/kaifeng/FYP/backup/sence_data.mat", mat_data)
        # print(states.flatten())
        # print(path)
        # print(pose.shape)
        # print(used_objects_index.flatten())
        # print(target_path.flatten())
        # print(target_path[0])
        # time.sleep(1000000)

    def _contact_expert(self):
        '''不断检测共享文件夹,检测是否获得专家经验'''
        while True:
            files = glob.glob(os.path.join('/home/kaifeng/FYP/docker_shared',"*"))
            output_file = [file for file in files if 'output.mat' in file]
            if len(output_file) != 0:
                print('---Successfully Get Expert Data------')
                break
            time.sleep(1)
            print("---Waiting for Expert---")

        with open(output_file[0], "r") as file:
            lines = file.readlines()
            steps = lines[-53:-3]
            data_list = [np.fromstring(row.strip("[]"), sep=" ") for row in steps]
            data_array = np.array(data_list)

        os.remove(output_file[0])

        return data_array




if __name__ == "__main__":

    '''
    env.PlaneID         : 0
    env.TableID         : 1
    env.robot.RobotID   : 2
    env.task.object_ids : [3, 4, ......]
    '''

    timer = Timer()

    env = RobotTaskEnv(render_mode='human',
                       time_sleep=False,
                       fixed_num_points=1024,
                       split = 0.5,
                       plane_points=0,
                       points_per_frame=256,
                       sub_steps=50,
                       debug_visulization=True,
                       num_objects=1,
                       )
    timer.reset()
    # 0.4S一个loop
    for i in range(1000000):
        print('\n---Step:{}---'.format(env._env_step))
        action = env.action_space.sample()
        # action[1] = 0
        # action[3] = 0
        # action[4] = 0
        # action[5] = 0
        # action = np.array([0, 0.3, 0, 0, 0, 0, 0])
        # action = np.array([0.3, 0, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 5 == 0:
            env.visualize_point_cloud()
        # for key in obs.keys():
        #     print(f"---key:{key}---shape:{obs[key].shape if isinstance(obs[key], np.ndarray) else obs[key]}---")
        #     time.sleep(5)

        #env.visualize_point_cloud(obs['all_PC'])
        #env.visualize_point_cloud()

        if terminated or truncated:
            env.reset()
        print('---Time Cost:{}---'.format(timer.record_and_reset()))
            
    p.disconnect()















    # timer = Timer()

    # env = RobotTaskEnv(render=True,
    #                    time_sleep=False,
    #                    fixed_num_points=1024,
    #                    split = 0.5,
    #                    plane_points=100,
    #                    points_per_frame=256,
    #                    sub_steps=50,
    #                    debug_visulization=True
    #                    )
    # obs = env.reset()
    # time.sleep(1)

    # for _ in range(1000000):
    #     expert_actions = env.get_expert_actions()

    #     for i in range(expert_actions.shape[0]):
    #         print('\n---Step:{}---'.format(env._env_step))
    #         obs, reward, terminated, truncated, info, expert_action = env.step_expert(expert_actions[i])
    #         print(reward,expert_action,'\n')

    #     env.visualize_point_cloud()
    #     env.reset()
        

    # p.disconnect()









