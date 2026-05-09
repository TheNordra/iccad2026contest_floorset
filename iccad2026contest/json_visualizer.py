import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

def visualize_viz_json(case_id):
    file_path = f"viz_results/case_{case_id}.json"
    
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    positions = data["positions"]
    block_types = data["block_types"]
    block_count = data["block_count"]
    
    COLOR_MAP = {
        0: '#4682B4',  # Steel Blue
        1: '#FF7F50',  # Coral
        2: '#3CB371'   # Medium Sea Green
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_aspect('equal', adjustable='box')
    
    # 用來追蹤所有方塊的邊界
    all_x_end = []
    all_y_end = []

    for i, (x, y, w, h) in enumerate(positions):
        b_type = block_types[i]
        c = COLOR_MAP.get(b_type, 'gray')
        
        # 建立矩形
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                 edgecolor='black', facecolor=c, alpha=0.8)
        ax.add_patch(rect)
        
        # 文字標籤
        txt_c = 'white' if b_type == 0 else 'black'
        ax.text(x + w/2, y + h/2, str(i), ha='center', va='center', 
                fontsize=8, fontweight='bold', color=txt_c)
        
        # 紀錄最大範圍
        all_x_end.append(x + w)
        all_y_end.append(y + h)

    # --- 關鍵修正：手動設定座標軸範圍 ---
    if all_x_end and all_y_end:
        max_x = max(all_x_end)
        max_y = max(all_y_end)
        min_x = min([p[0] for p in positions])
        min_y = min([p[1] for p in positions])
        # 加上 10% 的緩衝空間
        margin = max(max_x - min_x, max_y - min_y) * 0.05
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
    else:
        print("警告：沒有偵測到任何方塊座標資料")

    ax.set_aspect('equal')
    plt.title(f"Case {case_id}, Block Count: {block_count}")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    print(f"成功繪製 Test ID: {target_id}")
    plt.show()


if __name__ == "__main__":
    target_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    visualize_viz_json(target_id)