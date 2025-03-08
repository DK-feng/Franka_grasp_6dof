from gymnasium.envs.registration import register

register(
    id='FrankaGrasp6Dof-v0',  
    entry_point='franka_grasp_rl_6dof.core:RobotTaskEnv',  
    max_episode_steps=50, 
)