import pybullet as p
import numpy as np
import torch
import open3d as o3d
from torch_cluster import fps  
import time



def create_table(table_length: int=2,
                 table_width: int=1,
                 table_height: int=0.8,
                 table_position: list=[0,0,0],
                 table_color: list=[0.7, 0.4, 0.2, 1]):


    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[table_length / 2, table_width / 2, table_height / 2]
    )

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[table_length / 2, table_width / 2, table_height / 2],
        rgbaColor=table_color
    )

    table_position[2] += table_height/2
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=table_position
    )   

    return table_id





# 函数：使用随机采样或最远点采样对点云进行下采样或过采样
def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    规范化点云点数，如果点数多于 npoints,则下采样；如果点数少于 npoints,则过采样。
    use_farthest_point: 是否使用最远点采样(Farthest Point Sampling, FPS)进行下采样。
    """
    if pc.shape[0] == 0:
        print("\n---WARNING! Empty Point Cloud, Programme Stoped---\n")
        time.sleep(1000000)
        return pc
    elif pc.shape[0] > npoints:
        if use_farthest_point:
            pc_tensor = torch.from_numpy(pc).float()  # 转换为 Torch 张量
            fps_indices = fps(pc_tensor, ratio=npoints / pc.shape[0])  # 最远点采样
            pc = pc[fps_indices.numpy()]  # 选择采样后的点
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc