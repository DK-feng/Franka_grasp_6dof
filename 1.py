import glob
import os
import numpy as np
import scipy.io as sio
import glob
import time

# while True:
#     x = glob.glob(os.path.join('/home/kaifeng/FYP/docker_shared',"*"))
#     print(x)
#     time.sleep(1)

# path = os.path.join("/home/kaifeng/download_GADDPG/shared_data_grasping-20250213T202129Z-002/shared_data_grasping/objects","*")
# x = np.array(glob.glob(path))
# num_objects = np.random.randint(3,7)
# index = [np.random.randint(0,len(x)) for _ in range(num_objects)]
# file_path = x[index]

# print(file_path)

scene = sio.loadmat("/home/kaifeng/FYP/backup/sence_data.mat")
print(scene["used_objects_index"])
print(scene["pose"].shape)
print(scene["states"])
print(scene["target_name"])
print(scene["path"].shape)

# scene = sio.loadmat("/home/kaifeng/FYP/data/scenes/scene_2.mat")
# print(type(scene["path"]))
# print(type(scene["pose"]))
# print(type(scene["states"]))
# print(type(scene["target_name"]))

# with open('/home/kaifeng/FYP/docker_shared/output.mat', "r") as file:
#     lines = file.readlines()
#     steps = lines[-52:-2]
#     data_list = [np.fromstring(row.strip("[]"), sep=" ") for row in steps]
#     data_array = np.array(data_list)
#     print(data_array)
#     print(data_array.shape)

#     import numpy as np

