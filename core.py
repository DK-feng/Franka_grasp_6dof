import pybullet as p
from gymnasium import spaces
from typing import Optional
import numpy as np
import time
import open3d as o3d
from develop_utils import *
from utils import *
import glob
import os
import matplotlib.pyplot as plt
from scipy.io import savemat

'''
        点云转换到世界坐标系还是有问题，需要进一步调试
            外参没问题,主要是这个up_vector参数不知道干什么,将其强行设置为固定值[0,1,0]就会好很多,不知道原因
        Task的init和reset需要调整
        需要改正的地方搜索 “需要改正”
        删除重新载入物体会有视觉残留   :已解决    
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



class Panda:
    # robot class
    def __init__(self,
                 base_position: Optional[np.array] = (0, 0, 0),
                 image_size: Optional[np.array] = np.array([640, 480]),
                 debug_visulization: int = False):

        self.target_object_indice = 3
        self.base_position = base_position if base_position is not None else (0, 0, 0)
        self.ee_link = 11
        self.camera_link = 13
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.finger_indices = np.array([9, 10])
        self.joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
        #self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.neutral_joint_values = np.array([0.000, -1.048, 0.000, -2.010, 0.000, 1.543, 0.786, 0.04, 0.04])

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

        #垃圾参数
        self.loop = 0

    def _load_robot(self):
        self.RobotID = p.loadURDF("/home/kaifeng/FYP/URDF_files/panda_franka/panda_modified.urdf", self.base_position, useFixedBase=True)

    def reset(self):
        '''
        resetJointState 只改变状态，不控制电机
        p.resetJointState 是瞬时地改变关节状态，但它不会同步更新电机控制模式中的目标位置。
        如果你在仿真步进时仍然用 POSITION_CONTROL 控制，它会尝试回到之前的目标状态。
        POSITION_CONTROL 会覆盖 resetJointState 的效果
        如果 p.setJointMotorControlArray 在 reset() 后继续运行，电机会保持之前的目标位置。
        '''
        self._load_robot()
        self._set_joint_positions(self.neutral_joint_values)
        for _ in range(500):
            p.stepSimulation()
        for joint,angle in zip(self.joint_indices, self.neutral_joint_values):
            p.resetJointState(self.RobotID, joint, angle)

    def _process_mask(self,mask,target_object_id):
        #将mask中抓取目标物体设置为0，所有障碍物包括桌面设置为1， 背景设置为50
        mask[mask >= 0] += 1  # -1,1,2,3....
        mask[mask == target_object_id+1] = 0
        mask[mask > 0 ] = 1 #其余物体
        mask[mask == -1] = 50  #background
        return mask

    def render_from_camera(self):
        '''
        return:
            rgb_image:      np.array of shape (camera_height, camera_width, 3), 0-1
            depth_image:    np.array of shape (camera_height, camera_width, 1)
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
        width, height, rgb_image, depth_image, mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_image = np.array(rgb_image).reshape((self.camera_height, self.camera_width, 4))[:, :, :3]
        depth_image = np.array(depth_image).reshape((self.camera_height, self.camera_width))

        # 从非线性的 OpenGL 深度图(值为0-1)中提取真实的线性深度信息（实际距离）
        # depth_image = depth_image * (self.farVal - self.nearVal) + self.nearVal
        depth_image = self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depth_image)
        # depth_image = 2.0 * self.nearVal * self.farVal / (self.farVal + self.nearVal - (2.0 * depth_image - 1.0) * (self.farVal - self.nearVal))

        depth_image = depth_image[..., np.newaxis] #(480行,640列)即640宽*480高
        mask = np.array(mask).reshape((self.camera_height, self.camera_width))
        mask = self._process_mask(mask, self.target_object_indice)

        # Normalize
        rgb_image = rgb_image.astype(np.float32) / 255.0
        depth_image = depth_image.astype(np.float32)
        mask = mask.astype(np.float32)

        return rgb_image, depth_image, mask
    
    def _set_joint_positions(self, joint_positions: np.array):
        p.setJointMotorControlArray(self.RobotID,
                                    jointIndices=self.joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_positions,
                                    forces=self.joint_forces)
               
    def set_action(self, action: np.array):
        # action: postion change of end effector, (dx, dy, dz)
        target_position = p.getLinkState(self.RobotID, self.ee_link)[0] + action[:3]*0.05
        if target_position[2] < 0.02:
            target_position[2] = 0.02
        print("------Calculating Inverse Kinematics--- time cost:{:.3f}---".format(timer.record_and_reset()))
        target_orientation = p.getEulerFromQuaternion(p.getLinkState(self.RobotID, self.ee_link)[1]) + action[3:]*0.05
        target_orientation = p.getQuaternionFromEuler(target_orientation)
        joint_positions = p.calculateInverseKinematics(self.RobotID,
                                                        self.ee_link,
                                                        targetPosition=target_position,
                                                        targetOrientation=target_orientation)
        print("------Moving Joints--- time cost:{:.3f}---".format(timer.record_and_reset()))
        self._set_joint_positions(joint_positions)

    def set_joint_action(self, action: np.array):
        self._set_joint_positions(action)

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

    def get_camera_intrinsic(self):
        intrinsic_matrix = np.eye(3)
        # 假设 fov 为垂直视场角（单位：度）
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


class Task:
    # task class
    def __init__(self,
                 num_objects: int = 5,
                 use_urdf: bool = True):
        self.num_objects = num_objects
        self.objects_IDs = []
        self.objects_urdf_file_folder = '/home/kaifeng/FYP/data/objects'
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
        self.drop_positions = np.array([0.75, 0, 0.2])  #所有物体从这个高度自由落下形成structure cluster

        print("---Task Initialization: Done---------")

    def _load_objs(self):
        # 随即导入新物体,需要改正，参数仅适用于米老鼠
        objects_IDs = []
        select_objects_paths = np.random.choice(self.objects_files, size=self.num_objects, replace=False)
        if self.use_urdf:
            for file_path in select_objects_paths:
                obj_id = p.loadURDF(os.path.join(file_path,'model_normalized.urdf'),
                                    basePosition = self.drop_positions + np.array([np.random.uniform(-0.2, 0.2),
                                                                                    np.random.uniform(-0.2, 0.2),
                                                                                    np.random.uniform(-0.05, 0.05)]),
                                    baseOrientation = np.random.rand(4))
                print('------Loading Object:{}---ID:{}---'.format(file_path.split('/')[-1], obj_id))
                for _ in range(1000):
                    p.stepSimulation()
                objects_IDs.append(obj_id)
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

    def _set_target_object(self):
        target_object_id = np.random.choice(self.objects_IDs)
        return target_object_id

    def reset(self):
        """
        加载任务所需的物体，例如在堆积场景中加载几个 cube 模型
        """
        self.objects_IDs = self._load_objs()
        self.target_object_id = self._set_target_object()
        
    def compute_reward(self):
        """
        计算奖励函数
        """
        pass

    def is_success(self):
        """
        判断任务是否成功
        """
        return False

    def get_objects_info(self):
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
        target_object_path = self.select_objects_paths[self.objects_IDs.index(self.target_object_id)]
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



class RobotTaskEnv:
    # Main environment class
    def __init__(self, render=False,
                    background_color : Optional[np.array] = None,
                    timestep = 1/500,
                    sub_steps : int=20,
                    time_sleep : bool=False,
                    image_size: Optional[np.array] = np.array([640, 480]),
                    debug_visulization: bool = False,
                    pt_accumulate_ratio: float = 0.98,
                    fixed_num_points: int = 4096):

        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.background_color = background_color.astype(np.float32) / 255
        self.render = render
        self.timestep = timestep
        self.sub_steps = sub_steps 
        self.time_sleep = time_sleep
        self.image_size = image_size
        self.debug_visulization = debug_visulization

        self._env_step = 0
        self._pt_accumulate_ratio = pt_accumulate_ratio
        self._fixed_num_points = fixed_num_points

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
        self.task = Task()

        self.camera_intrinsic = self.robot.get_camera_intrinsic()

        # 6D action space: [dx, dy, dz, d_roll, d_pitch, d_yaw]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -np.pi/3, -np.pi/3, -np.pi/3]),
            high=np.array([1, 1, 1, np.pi/3, np.pi/3, np.pi/3]),
            dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.robot.camera_height, self.robot.camera_width, 3))

        self.curr_acc_points = np.zeros([0, 3])
        print("---RobotTaskEnv Initialization: Done---------")

    def _create_scene(self):
        self.PlaneID = p.loadURDF("/home/kaifeng/FYP/URDF_files/sense_file/floor/model_normalized.urdf", basePosition=[0,0,-0.5])
        self.TableID = create_table(table_length=1.8,
                                    table_width=1.2,
                                    table_height=0.5,
                                    table_position=[0.5,0,-0.5])

    def _reset_pybullet(self):
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timestep)

    def _get_obs(self):
        # 捕获初始摄像头点云并累计
        rgb_image, depth_image, mask = self.robot.render_from_camera()
        print("\n------Rendering--- time cost:{:.3f}---".format(timer.record_and_reset()))

        # [N, 3], 从depth_image里生成基于camera frame的目标物体的point_cloud
        new_pc_cameraframe = self._generate_pc_from_depth(depth_image, mask)

        # # 检查camera_to_world变换
        # array_x = np.array([i/6000*np.sin(x) for i,x in enumerate(range(2000))]).reshape(-1, 1)
        # array_y = np.array([i/6000*np.cos(x) for i,x in enumerate(range(2000))]).reshape(-1, 1)
        # array_z = np.linspace(0, 1, 2000).reshape(-1, 1)  # Create a 100x1 array with values from 0 to 1
        # new_pc_cameraframe = np.hstack([array_x, array_y, array_z])
        # print(new_pc_cameraframe.shape)
        # new_pc_worldframe = self._camera_to_world(new_pc_cameraframe)
        # self.visualize_point_cloud(new_pc_worldframe)
        # time.sleep(100000)

        # [N, 3], 将pc转换到world frame下 (已经检查为正确)
        new_pc_worldframe = self._camera_to_world(new_pc_cameraframe)
        print("---------Generating Point Clouds Based on Camera Frame--- time cost:{:.3f}---".format(timer.record_and_reset()))

        # [N + new points, 3], 先转换为世界坐标系下再累积
        self.update_curr_acc_points(new_pc_worldframe)
        print("---------Updating Culmulative Point Clouds--- time cost:{:.3f}---".format(timer.record_and_reset()))

        # 向上/下采样为固定点数，use_farthest_point = True的话非常耗时,0.7s一次
        self.curr_acc_points = regularize_pc_point_count(self.curr_acc_points, self._fixed_num_points, use_farthest_point=False)
        print("---------Regulization Point Clouds--- time cost:{:.3f}---".format(timer.record_and_reset()))
        obs = self.curr_acc_points
        
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
        self.robot.target_object_indice = self.task.target_object_id
        self.curr_acc_points = np.zeros([0, 3])

        obs = self._get_obs()

        return obs

    def _generate_pc_from_depth(self, depth_image, target_mask):
        # 从深度图像生成camera_frame的PC
        fx, fy = self.camera_intrinsic[0, 0], self.camera_intrinsic[1, 1]
        cx, cy = self.camera_intrinsic[0, 2], self.camera_intrinsic[1, 2]

        xs, ys = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]))
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)
        x_normalized = (xs - cx) / fx
        y_normalized = (ys - cy) / fy

        # all_points: (480, 640, 3)
        all_points = np.stack((x_normalized * -np.squeeze(depth_image),
                                y_normalized * -np.squeeze(depth_image),
                                -np.squeeze(depth_image)), axis=-1)
        masked_points = all_points[target_mask==0]
        camera_points = masked_points.reshape(-1, 3)
        return camera_points

    def _camera_to_world(self, point_clouds):
        '''将PC从camera frame转换为world frame'''
        # [4, N],最后一行为1
        camera_points_homogeneous = np.hstack((point_clouds, np.ones((point_clouds.shape[0], 1)))).T
        # view_matrix的逆矩阵就为外参
        view_matrix = np.array(self.robot.view_matrix).reshape(4, 4)
        camera_to_world = np.linalg.inv(view_matrix).T
        # introduction to robotics 19页
        world_points_homogeneous = camera_to_world @ camera_points_homogeneous
        # 注意形状变回(N, 3)
        world_points = world_points_homogeneous.T[:, :3]  # 只取前三个维度
        world_points[:,1] *= -1
        return world_points

    def update_curr_acc_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        # accumulate points
        index = np.random.choice(
            range(new_points.shape[0]),
            size=int(self._pt_accumulate_ratio**self._env_step * new_points.shape[0]),
            replace=False,
        ).astype(np.int32)
        self.curr_acc_points = np.concatenate(
            (new_points[index, :], self.curr_acc_points), axis=0)  

    def step(self, action):
        """
        执行一个动作，返回 observation, reward, done, info
        """
        for _ in range(self.sub_steps):
            self.robot.set_joint_action(action)
            p.stepSimulation()
            if self.time_sleep:
                time.sleep(self.timestep)
        print("---Excecuting Action:{}--- time cost:{:.3f}---".format(action, timer.record_and_reset()))

        obs = self._get_obs()
        reward = self.task.compute_reward()
        done = self.task.is_success()
        return obs, reward, done, {}

    def visualize_point_cloud(self, point_clouds=None):
        if point_clouds is None:
            point_clouds = self.curr_acc_points
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds)

        for i in range(100):
            print(point_clouds[i])

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
        rectangle_vertices = np.array([
            [0, -0.5, 0],
            [0, 0.5, 0],
            [1, 0.5, 0],
            [1, -0.5, 0]
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
            [pcd, world_frame, camera_frame, rectangle_line_set], 
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
        path, pose, target_path, used_objects_index = self.task.get_objects_info()
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
            steps = lines[-52:-2]
            data_list = [np.fromstring(row.strip("[]"), sep=" ") for row in steps]
            data_array = np.array(data_list)

        os.remove(output_file[0])

        return data_array

    def get_expert_actions(self):
        self._generate_mat_file()
        position_array = self._contact_expert()
        return position_array



if __name__ == "__main__":

    '''
    env.PlaneID         : 0
    env.TableID         : 1
    env.robot.RobotID   : 2
    env.task.object_ids : [3, 4, ......]
    '''

    timer = Timer()

    env = RobotTaskEnv(render=True,
                       time_sleep=True,
                       fixed_num_points=4096,
                       sub_steps=20,
                       debug_visulization=True,
                       )

    obs = env.reset()
    time.sleep(1)

    while True:
        expert_joint_states = env.get_expert_actions()
        print(expert_joint_states)
        print(expert_joint_states.shape)
        for action in expert_joint_states:
            obs, reward, done, _ = env.step(action)
        obs = env.reset()



    p.disconnect()

