import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm



def analyze_mask(mask: np.array):

    cmap = plt.get_cmap('tab10', np.max(mask) + 1)  # 颜色映射，确保每个值有唯一颜色
    norm = BoundaryNorm(np.arange(np.max(mask) + 2) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    img = ax.imshow(mask, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
    unique_values = np.unique(mask)  # 获取 mask 中所有唯一值
    patches = [mpatches.Patch(color=cmap(norm(val)), label=f'Value {val}') for val in unique_values]

    fig.canvas.draw()  # 强制 matplotlib 进行图形绘制，以确保所有的子图和元素（例如 ax）的位置已经计算完成
    right_x, top_y = ax.get_position().x1, ax.get_position().y1  # 右上角坐标
    legend_x = right_x + 0.2  # 偏移 0.2，确保 legend 不贴紧 mask
    ax.legend(handles=patches, bbox_to_anchor=(legend_x, top_y), loc='upper right' , title="Mask Values")
    ax.set_title("Mask Visualization with Different Colors")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()



if __name__ == '__main__':

    pass