import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def plot_floorplan(results, block_count, title="Floorplan Visualization"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 隨機生成顏色，方便區分不同的 Block
    def get_random_color():
        return (random.random(), random.random(), random.random())

    max_x = 0
    max_y = 0

    for i in range(block_count):
        x, y, w, h = results[i]
        
        # 繪製矩形
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', 
                                 facecolor=get_random_color(), alpha=0.6)
        ax.add_patch(rect)
        
        # 在矩形中間標上 Index
        plt.text(x + w/2, y + h/2, str(i), fontsize=8, ha='center', va='center')

        # 更新邊界以便調整視窗大小
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # 設定顯示範圍
    plt.xlim(0, max_x * 1.1)
    plt.ylim(0, max_y * 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

# 使用範例：
# 你可以在你的 MyOptimizer.solve 結束前，加上：
# self.plot_floorplan(results, block_count)