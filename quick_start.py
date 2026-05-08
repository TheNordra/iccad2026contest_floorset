import json
import os

def load_benchmark(file_path):
    if not os.path.exists(file_path):
        print(f"找不到檔案: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 讀取 Die 資訊 (邊界)
    die_width = data['die']['width']
    die_height = data['die']['height']
    
    # 讀取 Macros 資訊
    macros = data['macros']
    
    print(f"--- 數據解析成功 ---")
    print(f"Die 尺寸: {die_width} x {die_height}")
    print(f"Macro 總數: {len(macros)}")
    
    # 印出前兩個 Macro 看看
    for i, m in enumerate(macros[:2]):
        print(f"Macro[{i}] 名稱: {m['name']}, 面積: {m['area']}, 是否固定: {m.get('fixed', False)}")

if __name__ == "__main__":
    # 請根據你資料夾內的實際檔名修改路徑
    # 假設路徑在 iccad2026contest/benchmarks/範例檔.json
    target_file = "iccad2026contest/benchmarks/case1.json" 
    load_benchmark(target_file)