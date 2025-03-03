import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
import torch
import time

import torch.version


# 创建 PandaReach 任务环境
env = gym.make("PandaReach-v3")

# 由于 SAC 是 off-policy，n_envs 可以设置为 1
vec_env = make_vec_env(lambda: env, n_envs=1)

# 定义 SAC 训练参数
model = SAC(
    policy="MultiInputPolicy",       # 使用 MLP 策略（适用于 PandaReach）
    env=vec_env,
    batch_size=256,           # 批量大小（1024 太大，256-512 更合适）
    gamma=0.98,               # 折扣因子（更适合机器人任务）
    learning_rate=3e-4,       # 3e-4 是 SAC 默认学习率，适用于多数任务
    verbose=1,                # 输出训练信息
    train_freq=1,             # SAC 通常 train_freq=1 更稳定
    gradient_steps=1,         # 默认是 1，每个 step 训练一次
    tau=0.005,                # 软更新参数，默认 0.005
    buffer_size=1000000,      # 经验回放缓冲区（1M 经验更稳定）
    learning_starts=5000,     # 训练开始前收集 5000 经验
    replay_buffer_class=HerReplayBuffer,  # 使用 Hindsight Experience Replay (HER)
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    device="cuda:0",          # 使用 GPU 训练
    seed=0,                   # 设定随机种子，保证可复现
    tensorboard_log="./sac_panda_reach_tensorboard/"  # TensorBoard 日志
)

# 开始训练 1M step
model.learn(total_timesteps=100000)

# 保存模型
model.save("sac_panda_reach")
