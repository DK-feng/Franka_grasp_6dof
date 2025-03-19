import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import time
import torch

# class PointNet2(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space: gym.Space,
#         features_dim: int = 256,
#     ) -> None:
#         # [batch_size, 1024, 3]
#         super().__init__(observation_space, features_dim)
#         in_channel = 3
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(256, 40)

#     def forward(self, xyz):
#         B, _, _ = xyz.shape
#         norm = None #没有法向量
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))

#         # 屏蔽最后的全连接层
#         # x = self.fc3(x)
#         # x = F.log_softmax(x, -1)

#         # return x, l3_points
#         return x


class PointNet2(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
    ) -> None:
        # [batch_size, 3, 1024]
        super().__init__(observation_space, features_dim)
        in_channel = 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        norm = None #没有法向量
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        # 屏蔽最后的全连接层
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        # return x, l3_points
        return x


if __name__ == '__main__':

    # 0.06S

    DEVICE = torch.device('cuda')

    model = PointNet2()
    check_points = torch.load('/home/kaifeng/FYP/franka_grasp_rl_6dof/Extractor/PointNet2/checkpoints/best_model.pth')
    model.to(DEVICE)
    model.load_state_dict(check_points['model_state_dict'])
    model.eval()


    for i in range(1000):
        x = torch.randn([1, 3, 1024]).cuda()
        #x = torch.randn([1, 3, 1024])
        time_now = time.time()
        y = model(x)
        print(f'---epoch:{i}---time cost:{time.time() - time_now}---')
        #time.sleep()

        