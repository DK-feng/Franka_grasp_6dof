from franka_grasp_rl_6dof.core import RobotTaskEnv, Timer
import pybullet as p
import time
import numpy as np



if __name__ == '__main__':

    timer = Timer()

    env = RobotTaskEnv(render_mode='human',
                       time_sleep=True,
                       fixed_num_points=1024,
                       split = 0.5,
                       plane_points=0,
                       points_per_frame=256,
                       sub_steps=20,
                       debug_visulization=True,
                       num_objects=1,
                       )
    timer.reset()

    # 0.3S一个loop
    for i in range(1000000):
        print('\n---Step:{}---'.format(env._env_step))

        action = env.action_space.sample()
        # action = np.array([0.2, 0, -0.6, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        if (i+1) % 5 == 0:
            env.visualize_point_cloud()

        if terminated or truncated:
            time.sleep(1)
            env.reset()
        print('---Time Cost:{}---'.format(timer.record_and_reset()))
        print(f"reward:{reward}--terminated:{terminated}--truncated:{truncated}--time_step:{env._env_step}")

    p.disconnect()