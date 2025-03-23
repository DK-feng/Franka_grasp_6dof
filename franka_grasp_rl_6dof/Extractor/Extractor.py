from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict
from .PointNet2.pointnet2_cls_ssg import PointNet2
from .models import FCN
import numpy as np
import time

'''
    *   'all_PC':       all_PC,                             # (fixed_num_points, 3)
        'target_PC':    self.curr_acc_target_points,        # (fixed_num_points * split, 3)
        'obstacal_PC':  self.curr_acc_obstacal_points,      # (fixed_num_points * (1-split) - plane_points, 3)
        'plane_PC':     self.curr_acc_plane_points,         # (plane_points, 3)
    *   'timestep':     self._env_step,                     # 1~50
    *   'joint_state':  self.robot.get_joint_positions(),   # (7,)
    *   'ee_state':     ee_state                            # (x, y, z, rx, ry, rz, finger_width/2)

    
    self.extractors['concat_FCN']
    self.extractors['all_PC']
    self.projec_MLP
    self.fusion_layer
'''

class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        pointnet_output_dim: int = 256,
        fcn_output_dim: int = 128,
        fused_output_dim: int = 256,
        pointnet_weight_path: str = '/home/kaifeng/FYP/Extractor/PointNet2/checkpoints/best_model.pth',
        total_step: int = 30
    ) -> None:
        super().__init__(observation_space, features_dim=fused_output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = total_step

        # nn.ModuleDict，而不是一个普通的字典，确保子模块会被正确注册到模型中
        extractors: Dict[str, nn.Module] = {}

        concate_input_dim = observation_space.spaces['joint_state'].shape[0] + observation_space.spaces['ee_state'].shape[0]
        extractors['concat_FCN'] = FCN(
                spaces.Box(low=-2.0, high=2.0, shape=(concate_input_dim,), dtype=np.float32),
                fcn_output_dim).to(self.device)

        extractors['all_PC'] = PointNet2(
                observation_space.spaces["all_PC"],
                pointnet_output_dim).to(self.device)
        check_points = torch.load(pointnet_weight_path)
        extractors["all_PC"].load_state_dict(check_points['model_state_dict'])
        
        # 冻结 PointNet2 参数
        for param in extractors["all_PC"].parameters():
            param.requires_grad = False

        self.extractors = nn.ModuleDict(extractors)

        # timestep的投影MLP
        projec_dim = observation_space.spaces['ee_state'].shape[0] + observation_space.spaces['joint_state'].shape[0]
        self.projec_MLP = nn.Sequential(nn.Linear(1, projec_dim),
                                        nn.LayerNorm(projec_dim))

        # 融合所有数据
        self.fusion_layer = nn.Sequential(nn.LayerNorm(pointnet_output_dim + fcn_output_dim),
                                            nn.Linear(pointnet_output_dim + fcn_output_dim, fused_output_dim),
                                            nn.ReLU()).to(self.device)


    def forward(self, observations: TensorDict) -> torch.Tensor:
        past_time = time.time()
        # print('------------------------------')
        # print(f'all_PC shape:{observations["all_PC"].shape}')
        if len(observations["timestep"].shape) == 3:
            observations["timestep"] = observations["timestep"].squeeze(1)
        #print(f'timestep shape:{observations["timestep"].shape}')
        pc_features = self.extractors["all_PC"](observations["all_PC"].to(self.device).transpose(1,2))  # [batch_size, 3, 1024]

        normalized_timestep = (torch.argmax(observations['timestep'], dim=-1, keepdim=True) / (self.total_step - 1)).float()
        normalized_ee_state = (observations['ee_state'].float() + 4.0) / 8.0
        normalized_joint_state = (observations['joint_state'].float() + 4.0) / 8.0

        # print(f'normalized_timestep shape:{normalized_timestep.shape}')
        # print(f'normalized_ee_state:{normalized_ee_state.shape}')
        # print(f'normalized_joint_state:{normalized_joint_state.shape}')


        projected_timestep = self.projec_MLP(normalized_timestep)
        #print(f'projected_timestep:{projected_timestep.shape}')
        concat_input = (torch.cat((normalized_ee_state, normalized_joint_state), dim=-1) + projected_timestep).to(self.device)
        kinematics_features = self.extractors["concat_FCN"](concat_input).to(self.device)

        # print(f'pc_features shape:{pc_features.shape}')
        # print(f'kinematics_features:{kinematics_features.shape}')
        # print(f"extractor time cost:{time.time() - past_time}")
        # print(next(self.extractors['all_PC'].parameters()).device)
        # print(next(self.extractors['concat_FCN'].parameters()).device)
        # print(next(self.projec_MLP.parameters()).device)
        # print(next(self.fusion_layer.parameters()).device)
        # print('------------------------------\n')
        combined_features = torch.cat([pc_features, kinematics_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        return fused_features
    
