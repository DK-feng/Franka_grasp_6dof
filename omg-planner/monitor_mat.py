import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


# 共享目录路径（容器内部路径）
shared_dir = "/home/kaifeng/Projects/OMG-Planner/app/shared_dir"

# current working directory
cwd_dir = "/home/kaifeng/Projects/OMG-Planner/"

class MatFileHandler(FileSystemEventHandler):
    def on_created(self, event):

        if "output.mat" in event.src_path:
            return  # 直接跳过，不处理

        if event.src_path.endswith(".mat"):
            mat_path = event.src_path
            #/home/kaifeng/Projects/OMG-Planner/app/shared_dir/sence_data111.mat

            print(f"Detected new .mat file: {mat_path}")
            file_name = mat_path.split('/')[-1].split('.')[0]
            # 执行 bullet.panda_scene
            cmd = f"python3 -m bullet.panda_scene -f {file_name}"
            print(f"Executing: {cmd}")

            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd_dir)
                output = result.stdout.strip()
                print(f"Execution finished: {output}")

                # 将处理结果存入 output.mat
                output_file = os.path.join(shared_dir, "output.mat")
                with open(output_file, "w") as f:
                    f.write(output)

                # 删除 input.mat，防止重复处理
                os.remove(mat_path)
                print(f"Processed {mat_path}, waiting for next file.")

            except Exception as e:
                print(f"Error processing {mat_path}: {e}")

# 设置监听器
event_handler = MatFileHandler()
observer = Observer()
observer.schedule(event_handler, path=shared_dir, recursive=False)
observer.start()

print("Docker container is watching for .mat files...")
try:
    while True:
        observer.join(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
