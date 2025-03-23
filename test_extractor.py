from franka_grasp_rl_6dof.Extractor.Extractor import CustomCombinedExtractor
import time
import torch
from gymnasium import spaces
import numpy as np



observation_space = spaces.Dict(
    dict(
        all_PC=spaces.Box(-1.0, 1.0, shape=np.array([1024, 3]), dtype=np.float32),
        joint_state=spaces.Box(-4.0, 4.0, shape=7, dtype=np.float32), 
        ee_state=spaces.Box(-4.0, 4.0, shape=6, dtype=np.float32),
        timestep=spaces.Discrete(30)))



extractor = CustomCombinedExtractor(observation_space).to('cuda')

# 生成一批 dummy 数据（和真实 observation 结构一致）
batch_size = 32
dummy_observation = {
    'all_PC': torch.randn(batch_size, 1024, 3).to('cuda'),
    'ee_state': torch.randn(batch_size, observation_space.spaces['ee_state'].shape[0]).to('cuda'),
    'joint_state': torch.randn(batch_size, observation_space.spaces['joint_state'].shape[0]).to('cuda'),
    'timestep': torch.zeros(batch_size, 1, 30).to('cuda')  # one-hot
}

# 简单计时
start_time = time.time()
output = extractor(dummy_observation)
end_time = time.time()

print(f'处理时间: {(end_time - start_time) * 1000:.2f} ms')
