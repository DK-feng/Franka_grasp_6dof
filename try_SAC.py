import franka_grasp_rl_6dof
import gymnasium as gym
import time
from franka_grasp_rl_6dof.Extractor.Extractor import CustomCombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.logger import configure
from datetime import datetime



if __name__ == '__main__':

    env_id = 'FrankaGrasp6Dof-v0'
    num_cpu = 4
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"num_objects": 1,
                                    "render_mode": 'human'})


    model = SAC(policy="MultiInputPolicy",env=env, batch_size=1024, gamma=0.95, learning_rate=1e-4, verbose=1, 
            train_freq=32, gradient_steps=32, tau=0.05, tensorboard_log="./tmp", learning_starts=1500,
            buffer_size=50000, replay_buffer_class=HerReplayBuffer, device="cuda:0", seed=0,
            # Parameters for HER    
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            # Parameters for SAC
            policy_kwargs=dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(pointnet_weight_path = '/home/kaifeng/FYP/franka_grasp_rl_6dof/Extractor/PointNet2/checkpoints/best_model.pth'),
                net_arch=[512, 512, 512], 
                n_critics=2)
            )

    # print(model.policy)
    # time.sleep(10000)

    prefix = "test_sac"
    tmp_path = "./tmp/"+datetime.now().strftime(prefix + "_%H_%M_%d")
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=700_000, progress_bar=True)
    model.save(prefix + "_model")


