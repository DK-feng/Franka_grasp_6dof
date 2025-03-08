import franka_grasp_rl_6dof
import gymnasium as gym




env = gym.make('FrankaGrasp6Dof-v0', num_objects=1, render_mode='human')
observation, info = env.reset()


while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    print(info['env_step'])

env.close()
