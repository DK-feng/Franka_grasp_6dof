from franka_grasp_rl_6dof.core import RobotTaskEnv, Timer
import pybullet as p


if __name__ == '__main__':

    timer = Timer()

    env = RobotTaskEnv(render_mode='human',
                       time_sleep=False,
                       fixed_num_points=1024,
                       split = 0.5,
                       plane_points=0,
                       points_per_frame=256,
                       sub_steps=50,
                       debug_visulization=True,
                       num_objects=1,
                       )
    timer.reset()
    # 0.4S一个loop
    for i in range(1000000):
        print('\n---Step:{}---'.format(env._env_step))
        action = env.action_space.sample()
        # action[1] = 0
        # action[3] = 0
        # action[4] = 0
        # action[5] = 0
        # action = np.array([0, 0.3, 0, 0, 0, 0, 0])
        # action = np.array([0.3, 0, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 5 == 0:
            env.visualize_point_cloud()
        # for key in obs.keys():
        #     print(f"---key:{key}---shape:{obs[key].shape if isinstance(obs[key], np.ndarray) else obs[key]}---")
        #     time.sleep(5)

        #env.visualize_point_cloud(obs['all_PC'])
        #env.visualize_point_cloud()

        if terminated or truncated:
            env.reset()
        print('---Time Cost:{}---'.format(timer.record_and_reset()))
            
    p.disconnect()