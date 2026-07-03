import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys  # 匯入 sys 模組來讀取參數

def visualize_test_case(json_path, target_id):
    # 1. 讀取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {json_path}")
        return
    
    # 2. 尋找對應的 test_id
    results = data.get("test_results", [])
    target_case = next((item for item in results if item["test_id"] == target_id), None)
    
    if not target_case:
        print(f"錯誤：在 JSON 中找不到 test_id: {target_id}")
        return

    positions = target_case["positions"]
    cost = target_case.get("cost", 0)
    
    # 3. 繪圖設定 (同前)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal')
    
    max_x, max_y = 0, 0
    for i, (x, y, w, h) in enumerate(positions):
        random.seed(i) 
        color = [random.random() for _ in range(3)]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                 edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, str(i), fontsize=7, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    plt.title(f"Test Case {target_id} (Cost: {cost:.4f})")
    plt.xlim(-10, max_x + 50)
    plt.ylim(-10, max_y + 50)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # 設定預設檔案名稱
    # JSON_FILE = 'my_first_optimizer_results.json' 
    # JSON_FILE = 'optimizer_portfolio_results.json' 

    JSON_FILE = '../optimizer_constructive_results.json' 
    
    # 檢查終端機指令是否有帶參數
    if len(sys.argv) > 1:
        try:
            case_id = int(sys.argv[1]) # 取得指令後的第一個參數並轉為數字
            visualize_test_case(JSON_FILE, case_id)
        except ValueError:
            print("請輸入正確的數字編號，例如：python json_visualizer.py 10")
    else:
        # 如果沒帶參數，預設看第 0 個
        print("未偵測到參數，預設顯示 Test Case 0...")
        visualize_test_case(JSON_FILE, 0)