# ICCAD 2026 FloorSet — Session Context

## 環境

- **主程式**: `optimizer_claude.cpp` (C++) + `optimizer_claude.py` (Python wrapper)
- **Conda env**: `C:\Users\Nordra\.conda\envs\iccadv\python.exe`
- **Compiler**: `C:\msys64\ucrt64\bin\g++.exe`

### 編譯指令 (PowerShell)
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet"
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o optimizer_claude.exe optimizer_claude.cpp 2>&1
```

### 評估指令 (PowerShell)
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_claude.py 2>&1 | Select-Object -Last 12
```

### 分析指令
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_results.py
```

---

## 評分公式 (CRITICAL)

### 總分公式
```
Total Score = Σ Cost[i]·exp(n_i − max_n) / Σ exp(n_j − max_n)
```
- `max_n` = 最大 block 數（通常 120）
- **只有 n=118–120 的 case 才有顯著影響！** n=100 的 case 權重 ≈ 2e-9（幾乎 0）
- 改善平均分數沒意義，要專注在 top 3–4 個最大 case

### 個別 case 成本
```
Cost = (1 + 0.5*(HPWL_gap + Area_gap)) × exp(2*V_rel) × max(0.7, (rt/median_rt)^0.3)
```
- `HPWL_gap` = (your_HPWL - best_HPWL) / best_HPWL
- `Area_gap` = (your_area - best_area) / best_area
- `V_rel` = total_soft_violations / n_soft
- `rt/median_rt`: runtime 相對自己所有 case 的中位數，**全部 case 跑一樣長 → runtime_factor=1.0，無懲罰**
- **Hard timeout = 60 秒/case**

### Runtime 關鍵洞察
- 增加 TIME_LIMIT 不會有 runtime 懲罰（因為是自己的中位數做分母）
- 可以把重要的大 case 跑更久（最高 60 秒）

---

## Soft Constraint 類型

| 類型 | 說明 | 計算 |
|------|------|------|
| MIB | 同群組 block 必須相同形狀 | `n_mib = Σ(mib_size - 1)` |
| Cluster | 同群組 block 必須互相緊鄰（連通） | `n_cluster = Σ(cluster_size - 1)` |
| Boundary | block 必須接觸 bbox 邊/角 | `n_boundary = 每個 boundary block` |

```
n_soft = n_boundary + n_cluster + n_mib
V_rel = total_violations / n_soft
```

---

## optimizer_claude.cpp 目前狀態 (已編譯，待評估)

### 關鍵常數
```cpp
static const double TIME_LIMIT  = 8.00;   // SA 時間限制
static const double AREA_TOL    = 0.005;
static double W_VIOL            = 10.0;   // 初始值，SA 開始時動態校正
static double W_BOUNDARY        = 10.0;   // boundary distance gradient (固定)
static double W_AREA            = 0.05;   // 初始值，SA 開始時動態校正
static const int MAX_PACK_SIZE  = 60;     // cluster packing 最大 group 大小
static const int B_LEFT=1, B_RIGHT=2, B_TOP=4, B_BOTTOM=8;
```

### 動態 W_VIOL 校正 (run_sa() 中, SA 啟動後)
```cpp
double h0 = calc_hpwl_b2b(cur_pos) + calc_hpwl_p2b(cur_pos);
double a0 = calc_bbox_area(cur_pos);
if (a0 > 0 && h0 > 0) {
    W_AREA = h0 / a0;
    W_AREA = max(0.01, min(W_AREA, 2.0));
}
// 讓每個 violation ≈ 6× 比 1 unit HPWL gap 更重要（符合實際評分梯度）
if (h0 > 0 && n_soft > 0) {
    W_VIOL = max(50.0, h0 * 6.0);
}
```

**背景**: 原本 W_VIOL=10，而初始 HPWL ≈ 1000–5000，違反條件 term ≈ 5 vs HPWL term ≈ 1000–5000。SA 比例 200–400:1 偏向 HPWL，但實際評分違反條件的梯度比 HPWL 重要 6 倍。這是持續違反條件的根本原因。

### proxy_cost() 架構
```cpp
static double proxy_cost(const vector<Pos>& pos) {
    double c = calc_hpwl_b2b(pos) + calc_hpwl_p2b(pos)
             + calc_bbox_area(pos) * W_AREA
             + calc_violation(pos) * W_VIOL          // hard violations
             + calc_boundary_dist(pos) * W_BOUNDARY; // soft gradient
    // + aspect ratio penalty
}
```

### SA 移動類型 (8 種)
1. 隨機移動單一 block
2. 隨機旋轉單一 block
3. 交換兩個 block
4. 隨機 resize (soft block)
5. MIB sync resize
6. Cluster pack
7. 隨機翻轉
8. Boundary snap

### Post-processing: HPWL Hill-climber (slack_hpwl_opt)
SA 結束後執行，預算 0.15 秒：
- 對每個非 cluster、非固定的 block 計算 force（加權中心吸引）
- 沿主軸方向計算 slack（不重疊的最大移動距離）
- 只在 local HPWL 改善時接受移動
- 最多 40 pass，早停（若無移動）
- Boundary block 的 force 方向受限（不能離開邊界）

---

## 分數進展 (Total Score, 越低越好)

| 版本 | 分數 | 備註 |
|------|------|------|
| MAX_PACK_SIZE=12, 1.45s SA | 6.484 | 上一 session baseline |
| MAX_PACK_SIZE=60, 1.45s SA | 5.936 | 本 session baseline |
| +HPWL hill-climber, 1.45s SA | **5.885** | 目前最佳已評估 |
| 8s SA + hill-climber (無 W_VIOL 修) | 5.979 | 比 1.45s 差（n=119 variance）|
| 8s SA + dynamic W_VIOL | **4.5501** | n=120 case 99 cost=4.99 仍主導；viol 0.35–0.46 |
| + boundary_snap | **4.2173** | -7.3%; case 99 cost=4.29 |
| + cluster_snap (guarded) | **3.9557** | -6.2%; case 99 cost=4.02 |
| + MAX_PACK_SIZE=120 | **3.9481** | -0.2%; case 99 cost=4.17 |
| + cluster_snap 20 pass + axis retry | **3.9467** | -0.04% (邊際) |
| + W_VIOL × 2 (h0×12) | **3.8107** | -3.4%; case 99 cost=3.82; case 95 ↑ (variance) |
| W_VIOL ×3 (h0×18) | 3.8855 | 退步 |
| W_VIOL ×1.5 (h0×9) | **3.7944** | -0.4%（sweet spot） |
| W_VIOL ×1.25 (h0×7.5) | 4.0674 | 退步 |
| Multi-restart SA ×2 | 3.9717 | 每段時間過短，回到單次 SA |

---

## 下一步任務 (依優先序)

### 立即 (本 session 開始)
1. **評估目前版本** (8s SA + dynamic W_VIOL + hill-climber)
   - 預期大幅改善 violation 問題
   - 若 Total Score > 5.5，檢查 n=118–120 的 V_rel

### 短期優化方向

2. **Boundary snap post-processing**
   - SA 結束後，將 boundary block push 到最近邊緣
   - 條件：不產生 overlap
   - 預期：減少 boundary violations

3. **Cluster preplaced 相鄰擺放**
   - 若 cluster group 有 preplaced members，把 free members pack 在旁邊
   - 目前：free members 擺在 BL corner（可能離 preplaced 很遠）

4. **Multi-restart SA**
   - 在 TIME_LIMIT 內跑多次 SA（不同 seed），保留最佳
   - 減少 n=118–120 大 case 的 variance（這最影響分數）

5. **大 case 專用溫度策略**
   - n > 80 時降低初始溫度或加快冷卻
   - 因為大 case 的 SA 更不容易收斂

### 長期研究方向

6. **Sequence Pair (SP) 初始化**
   - 目前用 skyline BL packer，SP 可能給更好初始解
   
7. **精確 cluster 連通性修復**
   - 目前 cluster packing 保證連通，但若 MAX_PACK_SIZE 不夠大可能繞過 packing

---

## 已知問題 / 歷史 Bug

### slack_hpwl_opt 曾創造 overlap
- **原因**: 用 bounding box 做 y-overlap 檢查，漏掉部分 block
- **修法**: per-block slack 計算（對每個外部 block 單獨計算）

### Area gap 爆炸 (fixed)
- **原因**: bbox 包含了固定/preplaced block（在極端 TARGET 位置）
- **修法**: bbox 只從可移動 block 計算

### boundary force 方向錯誤 (fixed)
- **原因**: 程式碼壓制了朝向邊界的 force（反了）
- **修法**: `if ((bflag&B_LEFT) && fx>0) fx=0;`（壓制離開邊界的 force）

### W_VIOL 嚴重低估 (已修，待驗證)
- **原因**: W_VIOL=10 vs 初始 HPWL ≈ 1000–5000，比例差 200–400 倍
- **修法**: `W_VIOL = max(50.0, h0 * 6.0)`

---

## 檔案結構

```
FloorSet/
├── optimizer_claude.cpp    ← 主程式 (C++)
├── optimizer_claude.py     ← Python wrapper
├── optimizer_claude.exe    ← 編譯輸出
├── CLAUDE.md               ← 本檔案
└── iccad2026contest/
    ├── iccad2026_evaluate.py   ← 評估腳本
    ├── analyze_results.py      ← 分析腳本
    └── optimizer_claude_results.json  ← 評估結果
```

---

## analyze_results.py 輸出範例

當前已知問題 case（違反條件多）：
- Case 55: area_gap 過大（boundary block 在極端位置撐大 bbox）
- n=119 case: SA variance 大（隨機性強）

---

## 重要提醒

- **PowerShell 用分號或 `if ($?) {...}` 連接指令，不能用 `&&`**
- **評估需要 10–15 分鐘（100 個 case × 8 秒）**
- **只有 n=118–120 的 case 影響分數，不要被低 n case 的改善誤導**
- **W_VIOL 和 W_BOUNDARY 分開：W_VIOL 控制真實違反懲罰，W_BOUNDARY 是導引梯度**
