import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm



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


if __name__ == '__main__':

    print(0.98 ** 50)