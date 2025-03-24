import franka_grasp_rl_6dof
import gymnasium as gym
import time
from franka_grasp_rl_6dof.Extractor.Extractor import CustomCombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from datetime import datetime
import os




if __name__ == '__main__':

    # 记录地址
    prefix = "test_sac"
    timestamp = datetime.now().strftime("%H_%M_%d")
    experiment_name = f"{prefix}_{timestamp}"
    log_root = f"./logs/{experiment_name}"
    os.makedirs(log_root, exist_ok=True)

    env_id = 'FrankaGrasp6Dof-v0'
    num_cpu = 4

    # 训练环境
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"num_objects": 1,
                                    "render_mode": 'human'})
    
    # 评估环境
    eval_env = DummyVecEnv([lambda: gym.make(env_id, num_objects=1, render_mode='human')])
    eval_env.recording = False 

    # SAC + HER 配置
    model = SAC(policy="MultiInputPolicy",env=env, batch_size=128, gamma=0.98, learning_rate=1e-4, verbose=1, 
            train_freq=16, gradient_steps=16, tau=0.02, tensorboard_log=f"{log_root}/tensorboard", learning_starts=100,
            buffer_size=50000, device="cuda:0", seed=0,
            # Parameters for SAC
            policy_kwargs=dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(pointnet_weight_path = '/home/kaifeng/FYP/franka_grasp_rl_6dof/Extractor/PointNet2/checkpoints/best_model.pth'),
                net_arch=[512, 512, 512], 
                n_critics=2)
            )


    # # 视频录制配置
    # video_folder = f'{log_root}/videos/'
    # os.makedirs(video_folder, exist_ok=True)

    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     video_folder,
    #     record_video_trigger=lambda x: x % 4000 == 0,
    #     video_length=60,    # 两个episode
    #     name_prefix="6dof_eval")

    # logger
    new_logger = configure(log_root, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Callbacks, 每eval_freq次与环境交互后切换到eval_env，每save_freq次交互后自动保存
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{log_root}/best_model/",
        log_path=f"{log_root}/results/",
        eval_freq=4000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'{log_root}/checkpoints/',
        name_prefix=prefix
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    # 开始训练
    model.learn(total_timesteps=50_000, progress_bar=True, callback=callback)
    model.save(f"{log_root}/{prefix}_final_model")


