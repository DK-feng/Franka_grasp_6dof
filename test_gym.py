import franka_grasp_rl_6dof
import gymnasium as gym
import time



if __name__ == '__main__':

    env_id = 'FrankaGrasp6Dof-v0'
    env = gym.make(env_id, num_objects=1, render_mode='human')
    observation, info = env.reset()

    while True:
        time_now = time.time()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
        print(f"timestep:{info['env_step']}\t reward:{reward}\t time_cost:{time.time()-time_now}")


    env.close()
