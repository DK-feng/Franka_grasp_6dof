import pybullet as p
import numpy as np
import torch
import open3d as o3d
import time
import sys
from ..fps_cuda import fps_cuda




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


def farthest_point_sampling_gpu(point_cloud, num_samples):
    """
    Farthest Point Sampling (FPS) with GPU acceleration using PyTorch.
    
    Parameters:
        point_cloud (torch.Tensor): Input point cloud (N x 3), where N is the number of points.
        num_samples (int): Number of points to sample.
    
    Returns:
        sampled_points (torch.Tensor): The farthest sampled points (num_samples x 3).
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    point_cloud = torch.from_numpy(point_cloud).to(DEVICE)
    N = point_cloud.size(0)

    
    # Initialize variables
    sampled_points = torch.zeros(num_samples, 3, device=DEVICE)
    distances = torch.ones(N, device=DEVICE) * float('inf')
    
    # Randomly select the first point
    sampled_idx = torch.randint(0, N, (1,), device=DEVICE)
    sampled_points[0] = point_cloud[sampled_idx]
    
    for i in range(1, num_samples):
        # Compute the distance between each point and the nearest sampled point
        diff = point_cloud.unsqueeze(1) - sampled_points[:i].unsqueeze(0)  # (N, i, 3)
        dist = torch.norm(diff, dim=2)  # (N, i)
        
        # Find the minimum distance for each point
        min_distances, _ = torch.min(dist, dim=1)
        
        # Update the distance array
        distances = torch.minimum(distances, min_distances)
        
        # Select the farthest point
        farthest_idx = torch.argmax(distances)
        sampled_points[i] = point_cloud[farthest_idx]
        
    
    return sampled_points.cpu().numpy()


def farthest_point_sampling_cuda(point_cloud, num_samples):
    # 输出:(N, 3),不支持batch纬度
    assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3
    point_cloud = torch.from_numpy(point_cloud).float().cuda().unsqueeze(0)
    pc_1024 = fps_cuda.farthest_point_sampling(point_cloud, num_samples)
    return point_cloud.squeeze(0)[pc_1024.squeeze(0)].cpu().numpy()


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    规范化点云点数，如果点数多于 npoints,则下采样；如果点数少于 npoints,则过采样。
    use_farthest_point: 是否使用最远点采样(Farthest Point Sampling, FPS)进行下采样。
    """
    npoints = int(npoints)
    if pc.shape[0] == 0:
        print("\n---WARNING! Empty Point Cloud---\n")
        return pc
    
    elif pc.shape[0] > npoints:
        if use_farthest_point:
            pc = farthest_point_sampling_cuda(pc, npoints)  # 最远点采样
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def get_target_obj_from_mask(mask):
    '''选择mask中面积最大的值作为目标,除去背景'''
    unique_values, counts = np.unique(mask, return_counts=True)
    sorted_indices = np.argsort(-counts)
    max_value = unique_values[sorted_indices[0]]    #选择最多的
    if max_value == 0:
        max_value = unique_values[sorted_indices[1]]    #如果最多为背景则选择第二多的
    return max_value


def process_mask(mask, target_mask_value):
    #将mask中抓取目标物体设置为0，所有障碍物设置为1，桌面和背景设置为50
    if target_mask_value is None:
        return None
    current_mask = mask.copy()
    current_mask[current_mask == target_mask_value] = 666
    current_mask[current_mask == 0 ] = 50   #桌面和背景
    current_mask[(current_mask != 666) & (current_mask != 50)] = 1  #障碍物
    current_mask[current_mask == 666] = 0  #目标物体
    return current_mask


def bound_points(points, centroid, width):
    '''将点云点限制在一个边界内'''
    new_points = points.copy()
    new_points = new_points[
        (centroid[0]-width < new_points[:, 0]) & (new_points[:, 0] < centroid[0]+width) &
        (centroid[1]-width < new_points[:, 1]) & (new_points[:, 1] < centroid[1]+width)
    ]
    return new_points


def normalize_point_cloud(points):
    """
    对点云数据进行归一化：
    1. 使点云中心化（零均值）
    2. 归一化到单位球（单位半径）

    参数:
    - points: (N, 3) 的 numpy 数组，表示 N 个三维点

    返回:
    - 归一化后的点云 (N, 3)
    """
    # 计算质心（center of mass）
    centroid = np.mean(points, axis=0)

    # 使点云中心化
    points -= centroid

    # 计算所有点的最大范数（L2 距离）
    max_norm = np.max(np.linalg.norm(points, axis=1))

    # 归一化到单位球
    points /= max_norm

    return points







if __name__ == '__main__':
    import fps_cuda


    pc = np.random.randn(2048, 3)
    print(pc.shape)
    current_time = time.time()
    #pc_1024 = farthest_point_sampling_gpu(pc, 1024)
    # pc_1024 = farthest_point_sampling_torch_optimized(pc, 1024)\
    pc = torch.from_numpy(pc).float().cuda()
    pc_1024 = fps_cuda.farthest_point_sampling(pc, 1024)
    print(f'---{time.time() - current_time}---') 