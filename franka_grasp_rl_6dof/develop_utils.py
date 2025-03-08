import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import open3d as o3d


def analyze_mask(mask: np.array):
    mask = mask.astype(int)
    unique_values = np.unique(mask)  # 获取 mask 中所有唯一值
    
    # 生成随机颜色，并确保不同值颜色不同
    np.random.seed(42)  # 固定随机种子以获得稳定颜色
    colors = np.random.rand(len(unique_values), 3)  # 生成 RGB 颜色
    cmap = ListedColormap(colors)  # 创建自定义 colormap
    
    # 生成 BoundaryNorm，确保颜色正确映射
    norm = BoundaryNorm(np.arange(len(unique_values) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    img = ax.imshow(mask, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')

    # 创建颜色-值映射
    color_dict = {val: colors[i] for i, val in enumerate(unique_values)}
    patches = [mpatches.Patch(color=color_dict[val], label=f'Value {val}') for val in unique_values]

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Mask Values")
    ax.set_title("Mask Visualization with Correct Colors")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()



def visualize_point_cloud(point_clouds):
    # 可视化点云

    all_pcd = {}
    for i,point_cloud in enumerate(point_clouds):
        all_pcd[str(i)] = o3d.geometry.PointCloud()
        all_pcd[str(i)].points = o3d.utility.Vector3dVector(point_cloud)
        color = [1,0,0] if i==0 else ([0,0,1] if i==1 else ([0,1,0] if i==2 else np.random.rand(3)))
        all_pcd[str(i)].paint_uniform_color(color) 
    pcds = list(all_pcd.values())

    # 世界坐标系原点
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 创建矩形顶点
    width = 0.4
    rectangle_vertices = np.array([
       [-0.5, 0],
       [-0.5, 1],
       [0.5, 1],
       [0.5, 0]
    ])
    # 创建矩形边的索引
    rectangle_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]]

    # 创建 LineSet 对象
    rectangle_line_set = o3d.geometry.LineSet()
    rectangle_line_set.points = o3d.utility.Vector3dVector(rectangle_vertices)
    rectangle_line_set.lines = o3d.utility.Vector2iVector(rectangle_lines)

    # 可视化点云和坐标系  
    o3d.visualization.draw_geometries(
        pcds + [world_frame, rectangle_line_set], 
        window_name="Point Cloud with Multiple Coordinate Frames",
        width=1280, height=960)


if __name__ == '__main__':

    print(0.98 ** 50)