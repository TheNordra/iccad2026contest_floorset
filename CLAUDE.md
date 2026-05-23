# ICCAD 2026 FloorSet — Session Context

## ⚠️ 重要變更通知

**官方評分系統已變更。** 本檔案目前記錄的分數與優化策略，是基於 **舊版** 評分公式
所推導與調整的。請在繼續任何優化前，**先確認新版評分公式**，並重新評估：

1. 各參數權重（W_VIOL、W_AREA、W_BOUNDARY）是否仍適用
2. Post-processing 順序與內容是否仍有效
3. 哪些 case 是新的高權重 case（不一定還是 n=118–120）

---

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

### 單 case 快速測試 (PowerShell)
```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_claude.py --test-id 99
```

### 分析指令
```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_results.py
```

---

## 目前狀態 (Current Status)

### 最佳已驗證版本：Total Score = **3.7944** (舊版評分)
- **TIME_LIMIT = 8.00 秒**
- **W_VIOL = h0 × 9** (即舊基準 ×1.5)
- **W_AREA = h0/a0 (clamped to [0.01, 2.0])** — 動態校正
- **W_BOUNDARY = 10.0** (固定，作為梯度引導)
- **MAX_PACK_SIZE = 120** — 所有 cluster 都用 multirow 結構化 packing

### 程式碼狀態 (`optimizer_claude.cpp`)
> ⚠️ 注意：最後一次失敗實驗 (TIME_LIMIT=12) 的修改尚未回退，
> 請編譯前確認 `TIME_LIMIT = 8.00`。

關鍵組件（依執行順序）：

1. **初始 permutation**：connectivity-driven order，再依優先序重排：
   - 0: cluster blocks（按 gid 連續）
   - 1: LEFT/BOTTOM boundary blocks
   - 2: 一般 blocks
   - 3: RIGHT/TOP boundary blocks

2. **Skyline decode**：
   - `pack_cluster_multirow`: ceil(sqrt(nm)) 行寬，最高塊靠左以保證行間相接
   - `pack_cluster_anchored`: cluster 有 preplaced member 時，pack 在 preplaced 旁
     （目前 SA 中啟用，但歷史測試顯示會擴大 area，需重新評估在新評分下的影響）

3. **W_VIOL/W_AREA 動態校正**（SA 開始時）：
   ```cpp
   W_AREA = clamp(h0/a0, 0.01, 2.0);
   W_VIOL = max(50.0, h0 * 9.0);  // 1.5× from sweep
   ```

4. **SA 主迴圈 (8 秒)**：8 種 move 類型
   - 0.30 swap / 0.14 relocate / 0.08 connectivity move
   - 0.10 resize aspect ratio / 0.08 rotate
   - 0.12 MIB unify / 0.08 cluster adjacency / fallback double-swap
   - 時間導向冷卻：T = 200 × (0.05/200)^(elapsed/TIME_LIMIT)

5. **Post-processing 序列**（總預算 ~0.15s）：
   - `cluster_snap`: 將斷開的 cluster member 朝 anchor 滑動（guarded by violation check, 20 passes, axis fallback）
   - `boundary_snap`: 將 boundary block 推向所需邊緣
   - `slack_hpwl_opt`: 對非 cluster block 做 HPWL-improving 滑動
   - `boundary_snap`（再次）：修正 hill-climber 的偏移

---

## 分數歷程 (舊評分公式下)

| 版本 | 分數 | 備註 |
|------|------|------|
| MAX_PACK_SIZE=12, 1.45s SA | 6.484 | 上一 session baseline |
| MAX_PACK_SIZE=60, 1.45s SA | 5.936 | session 起點 |
| +HPWL hill-climber | 5.885 | |
| 8s SA + dynamic W_VIOL (×1.0) | 4.5501 | n=120 主導 |
| + boundary_snap | 4.2173 | -7.3% |
| + cluster_snap (guarded) | 3.9557 | -6.2% |
| + MAX_PACK_SIZE=120 | 3.9481 | -0.2% |
| + cluster_snap 20 pass + axis retry | 3.9467 | 邊際 |
| + W_VIOL ×2 (h0×12) | 3.8107 | -3.4% |
| W_VIOL ×3 (h0×18) | 3.8855 | 過頭 |
| **W_VIOL ×1.5 (h0×9)** | **3.7944** | **當前最佳** |
| W_VIOL ×1.25 (h0×7.5) | 4.0674 | 退步 |
| Multi-restart ×2 (4s+4s) | 3.9717 | 每段 SA 時間不足 |
| TIME_LIMIT=12s | 4.3120 | 意外退步（SA 過熱階段太長） |

---

## 這個階段想解決的問題

### A. 評分公式變更後的重新校正
1. **取得新評分公式**：閱讀更新後的 `iccad2026_evaluate.py` / `cost.py`，
   確認以下項目是否改變：
   - Total Score 加權方式（是否仍以 exp(n_i - max_n) 加權？）
   - 個別 cost 公式（HPWL_gap、Area_gap、V_rel、runtime_factor 的係數）
   - 違反條件定義（boundary touch tolerance、cluster connectivity 判斷）
   - 是否新增其他懲罰項
2. **重新識別高權重 case**：新評分下哪些 case 真正影響 total？
3. **重新校準 proxy_cost 權重**：W_VIOL、W_AREA、W_BOUNDARY

### B. 在最佳版本（3.7944）基礎上續優化的瓶頸
即使在舊評分下，top case 仍有以下問題未解：
- HPWL gap 普遍 0.7–1.5（明顯高於最佳解）
- Area gap 0.3–0.8
- V_rel 0.3–0.45（cluster 與 boundary 仍有殘留違反）

可能原因：
- SA 8s 不足以收斂 n≥110 的大 case
- post-processing 的 slack 限制無法做大幅移動
- cluster 連通性修復受限於 calc_violation guard（過於保守）

### C. 未解決的歷史問題
- **case 95 (n=116) 變異大**：W_VIOL 提升時其 viol 反而上升，可能是 SA 卡在不同 basin
- **cluster preplaced anchored pack 副作用**：擴大 area gap（目前已在程式碼中，但效果中性偏負）
- **runtime_factor 的影響**：所有 case 跑 8s 時 factor=1.0，但若新評分下時間策略不同，需重新設計

---

## 預期目標

### 短期 (1–2 個迭代)
- **目標 1**：在新評分系統下達到「同等於或優於 3.79 在舊評分下的相對位置」
- **目標 2**：理解新評分對 violation / HPWL / area 三者的相對權重，
  並調整 W_VIOL、W_AREA 至最佳比例
- **目標 3**：建立新評分下的 baseline 與 case-level 分析（必須有新的
  analyze_results.py 來顯示 top cases 與其分項貝獻）

### 中期 (3–5 個迭代)
- **目標 4**：將 top case (依新評分) 的 cost 降至接近中位數的 1.3 倍以內
- **目標 5**：穩定性提升 — 多次評估的 score 變異 ≤ 2%
  （目前單次 SA 變異可達 5–10%）
- **目標 6**：若 runtime_factor 在新評分下允許較長時間，重新評估
  `TIME_LIMIT` 與 multi-restart 的可行性

### 長期 (重大架構改變)
- **目標 7**：替換 skyline BL packer 為更強的初始解產生器
  （見「未來發展方向」第 1 點）
- **目標 8**：將 SA 中的 violation handling 從「penalty in proxy_cost」改為
  「constraint repair after each move」，避免 SA 卡在違反區域

---

## 未來發展方向

### 1. 更強的初始解產生器
- **Sequence Pair (SP)** 編碼取代 perm + skyline BL
  - 優點：可表達非 BL-packing 的解空間，搜尋空間更廣
  - 缺點：移動定義較複雜，重新編碼成本高
- **B*-Tree**：另一種主流 floorplan 編碼，連續滑動更直觀
- **Force-Directed Initialization**：用 b2b/p2b 的力場做 quadratic placement
  作為初始解，再用 BL packer 解 overlap

### 2. SA 演進
- **Adaptive cooling**：根據 acceptance rate 動態調整冷卻速度
- **Restart with perturbation**：偵測 stagnation 時對 best_pos 做局部擾動
  再繼續（不是 cold restart）
- **Move 機率自適應**：成功率高的 move 類型給予更高機率
- **大 case 專屬策略**：n≥100 時用更慢的冷卻、更多 cluster move

### 3. 更深入的 Post-processing
- **Cluster rigid translation**：將整個 cluster 當作剛體向 boundary / 連通鄰居移動
- **Bbox shrinking**：對遠離 bbox 邊的非 boundary block 做反向 push
  （減少 area_gap）
- **MIB shape repair**：post-process 統一 MIB group 內所有成員的 (w, h)
- **2-axis hill-climber**：目前只在 dominant axis 移動，可改為 X-then-Y 序列移動
  （需小心 corner 與 overlap）

### 4. 結構化的 violation 修復
- 把 calc_violation 分解成 boundary / cluster / mib 三項，分別統計與輸出
  → 可清楚知道哪類 violation 主導，集中修復
- 對 cluster 違反：BFS 找最大 component，將其他 component 朝它做 group 移動
- 對 boundary 違反：考慮**移動 bbox 內部的 block 來縮 bbox**，而不是只
  移動 boundary block 自身

### 5. 工具與分析
- **修復 analyze_violations.py**：目前 lite_dataset_test 模組缺失，無法跑
- **建立分項貝獻表**：score = Σ weighted_cost，列出每個 top case 的
  HPWL/Area/Viol/Runtime 分項對 total 的 contribution（百分比）
- **regression suite**：保留每個里程碑版本的 `optimizer_claude_results.json`，
  以便快速 diff（目前只有最新一份）

### 6. 效能優化（若 TIME_LIMIT 仍受限）
- calc_violation 目前 O(N² + Σ cluster_size²)，可用 spatial index 加速
- skyline_decode 每個 SA move 都重新呼叫，可改 incremental update
- post-processing 的 violation guard 是熱點，可只重新計算受影響 cluster

---

## 已知 Bug / 注意事項

- **PowerShell 用分號或 `if ($?) {...}` 連接指令，不能用 `&&`**
- **評估需要 13–15 分鐘（100 個 case × 8 秒）**
- **`optimizer_claude.cpp` 中 `TIME_LIMIT` 可能殘留 12.00**（最後一次失敗實驗），
  繼續優化前請確認為 8.00
- **`analyze_violations.py` 無法執行**（lite_dataset_test 缺失）
- **slack_hpwl_opt 曾創造 overlap**（已修：per-block slack 計算）
- **bbox 包含 preplaced 導致 area gap 爆炸**（已修：bbox 只從可移動 block 計算）
- **boundary force 方向錯誤**（已修：壓制離開邊界的 force）

---

## 檔案結構

```
FloorSet/
├── optimizer_claude.cpp    ← 主程式 (C++)
├── optimizer_claude.py     ← Python wrapper
├── optimizer_claude.exe    ← 編譯輸出
├── CLAUDE.md               ← 本檔案
└── iccad2026contest/
    ├── iccad2026_evaluate.py   ← 評估腳本（可能已隨新評分更新）
    ├── analyze_results.py      ← 分析腳本（可用）
    ├── analyze_violations.py   ← violation 分項（壞掉，待修）
    ├── check_viols.py          ← 同上
    └── optimizer_claude_results.json  ← 最新評估結果
```

---

## 給下一個 session 的優先建議

1. **第一件事**：閱讀 `iccad2026contest/iccad2026_evaluate.py` 與 `cost.py`，
   確認新評分公式
2. **第二件事**：把 `TIME_LIMIT` 確認為 8.00 並重新編譯，跑一次 baseline
3. **第三件事**：依新評分跑 analyze_results.py，找出新的 top cases 與其分項
4. **再來才是**：依「這個階段想解決的問題」展開實驗
