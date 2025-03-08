import torch
from .lib import networks
from .tools.test_images import compute_xyz
from .lib.fcn.test_dataset import test_sample
import numpy as np
import time



class UCN:
    'UNseen Clustering Network'
    def __init__(self, camera_intrinsic, NUM_CLASS: int=2, DEVICE=torch.device('cuda:0'), PIXEL_MEANS=None):
        self.NUM_CALSS = NUM_CLASS
        self.DEVICE = DEVICE
        self.network_name = 'seg_resnet34_8s_embedding'
        self.network_data = torch.load('franka_grasp_rl_6dof/UCN/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth')
        self.network = networks.__dict__[self.network_name](NUM_CLASS, 64, self.network_data).cuda(device=DEVICE)
        self.network.eval()
        # self.network_data_crop = torch.load('UCN/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth')
        # self.network_crop = networks.__dict__[self.network_name](NUM_CLASS, 64, self.network_data_crop).cuda(device=DEVICE)
        self.network_crop = None
        # self.network_crop.eval()

        self.PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717]).reshape(1,1,3) if PIXEL_MEANS is None else PIXEL_MEANS
        self.camera_intrinsic = camera_intrinsic

        self.fx = self.camera_intrinsic[0, 0]
        self.fy = self.camera_intrinsic[1, 1]
        self.px = self.camera_intrinsic[0, 2]
        self.py = self.camera_intrinsic[1, 2]

    def get_mask(self, rgb_image, depth_image, num_objects):
        '''只返回mask,返回的refined mask为None'''

        # rgb
        rgb_image = torch.from_numpy(rgb_image)
        pixel_means = torch.from_numpy(self.PIXEL_MEANS / 255.0)
        rgb_image -= pixel_means
        rgb_image = rgb_image.permute(2, 0, 1)
        data = {'image_color': rgb_image.unsqueeze(0)}

        # depth
        height, width = depth_image.shape[0], depth_image.shape[1]
        point_cloud = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, height, width)
        depth_data = torch.from_numpy(point_cloud).permute(2, 0, 1)
        data['depth'] = depth_data.unsqueeze(0)
        out_label, out_label_refined = test_sample(data, self.network, self.network_crop, device=self.DEVICE, num_objects=num_objects)
        return out_label, out_label_refined





def UCN_get_mask(rgb_image,
                depth_image,
                camera_intrinsic,
                num_objects,
                DEVICE=torch.device('cuda:0'),
                NUM_CLASS: int=2,):
    '''返回位置物体的场景分割'''
    network_name = 'seg_resnet34_8s_embedding'

    # 加载预训练模型
    network_data = torch.load('UCN/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth')
    network = networks.__dict__[network_name](NUM_CLASS, 64, network_data).cuda(device=DEVICE)
    network.eval()
    network_data_crop = torch.load('UCN/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth')
    network_crop = networks.__dict__[network_name](NUM_CLASS, 64, network_data_crop).cuda(device=DEVICE)
    network_crop.eval()


    # rgb
    PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717]).reshape(1,1,3)  # 以 BGR 顺序为例
    rgb_image = torch.from_numpy(rgb_image)
    pixel_means = torch.from_numpy(PIXEL_MEANS / 255.0)
    rgb_image -= pixel_means
    rgb_image = rgb_image.permute(2, 0, 1)
    data = {'image_color': rgb_image.unsqueeze(0)}

    # depth
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    px = camera_intrinsic[0, 2]
    py = camera_intrinsic[1, 2]
    height, width = depth_image.shape[0], depth_image.shape[1]
    point_cloud = compute_xyz(depth_image, fx, fy, px, py, height, width)
    depth_data = torch.from_numpy(point_cloud).permute(2, 0, 1)
    data['depth'] = depth_data.unsqueeze(0)

    out_label, out_label_refined = test_sample(data, network, network_crop, device=DEVICE, num_objects=num_objects)

    return out_label, out_label_refined



