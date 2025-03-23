import pybullet as p
import pybullet_data
import time

# 启动 DIRECT 模式
cid = p.connect(p.DIRECT)


# 加载场景
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("r2d2.urdf", [0, 0, 1])

# 相机参数
width, height = 640, 480
fov, aspect, near, far = 60, width / height, 0.1, 100
viewMatrix = p.computeViewMatrix(cameraEyePosition=[1, 1, 1],
                                 cameraTargetPosition=[0, 0, 0],
                                 cameraUpVector=[0, 0, 1])
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# 连续渲染几帧，测试性能
for i in range(5):
    t1 = time.time()

    img = p.getCameraImage(width,
                           height,
                           viewMatrix,
                           projectionMatrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)

    t2 = time.time()
    print(f"✅ Frame {i} rendered in {t2 - t1:.4f} seconds. RGB shape: {len(img[2])}")

# 释放资源
p.disconnect()
