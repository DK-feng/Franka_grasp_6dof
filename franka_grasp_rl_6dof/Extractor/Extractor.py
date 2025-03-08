from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict
from Extractor.PointNet2.pointnet2_cls_ssg import PointNet2
from Extractor.models import FCN


'''
    *   'all_PC':       all_PC,                             # (fixed_num_points, 3)
        'target_PC':    self.curr_acc_target_points,        # (fixed_num_points * split, 3)
        'obstacal_PC':  self.curr_acc_obstacal_points,      # (fixed_num_points * (1-split) - plane_points, 3)
        'plane_PC':     self.curr_acc_plane_points,         # (plane_points, 3)
    *   'timestep':     self._env_step,                     # 1~50
    *   'joint_state':  self.robot.get_joint_positions(),   # (7,)
    *   'ee_state':     ee_state                            # (x, y, z, rx, ry, rz, finger_width/2)
'''

class CustomCombinedExtractor(BaseFeaturesExtractor):
    '''
    all_PC  >>>   PointNet2    >>>    256
    [timestep, joint_state, ee_state]   >>>   concate   >>>    concate_FCN   >>>   128
    '''
    def __init__(
        self,
        observation_space: spaces.Dict,
        pointnet_output_dim: int = 256,
        fc_output_dim: int = 128,
        pointnet_weight_path: str = '/home/kaifeng/FYP/Extractor/PointNet2/checkpoints/best_model.pth'
    ) -> None:
        super().__init__(observation_space, features_dim=pointnet_output_dim+fc_output_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.extractors: Dict[str, nn.Module] = {}

        # 1是timestep
        concate_input_dim = 1 + observation_space.spaces['joint_state'].shape[0] + observation_space.spaces['ee_state'].shape[0]
        self.extractors['concat_FCN'] = FCN(
                spaces.Box(low=-5.0, high=5.0, shape=(concate_input_dim,), dtype=torch.float32),
                fc_output_dim).to(self.device)

        self.extractors['all_PC'] = PointNet2(
                observation_space.spaces["all_PC"],
                pointnet_output_dim).to(self.device)
        self.extractors["all_PC"].load_state_dict(torch.load(pointnet_weight_path, map_location=self.device))
        
        # 冻结 PointNet2 参数
        for param in self.extractors["all_PC"].parameters():
            param.requires_grad = False

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        encoded_tensor_list.append(self.extractors["all_PC"](observations["all_PC"].to(self.device)))

        concat_input = torch.cat((observations["timestep"].float().unsqueeze(-1),
                                    observations["joint_state"],
                                    observations["ee_state"]), dim=-1).to(self.device)

        encoded_tensor_list.append(self.extractors["concat_FCN"](concat_input))

        return torch.cat(encoded_tensor_list, dim=1)
    