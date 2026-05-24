# ICCAD 2026 FloorSet — Session Context

## 新評分公式 (已確認, 2026-05-23)

### 公式本身
- **Cost** (per case): `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) × exp(2*V_rel) × max(0.7, R^0.3)`
  - 不可行 `cost = 10.0`；feasible 上限 `9.999999`
  - `HPWL_gap` / `Area_gap` 從下方 clamp 到 0（超過 baseline 沒額外獎勵）
  - `V_rel = (V_boundary + V_grouping + V_mib) / N_soft`，N_soft = boundary blocks + Σ(MIB-1) + Σ(Cluster-1)
- **Total Score**: `Σ Cost[i] · exp(n_i/12) / Σ weight`
  - 與舊版 (`exp(n_i - max_n)`) 比較：
    - 舊版 n=120 一案佔 ~63%
    - 新版 n=120 佔 ~8%，n>=110 累計 ~53%（小/中型 case 變得遠比之前重要）

### 評分權重表（新版）
| n | 權重 e^(n/12) | 佔比 |
|---|---|---|
| 120 | 22026 | 8.0% |
| 110 | 8939 | 3.2% |
| 100 | 4143 | 1.5% |
| 90 | 1808 | 0.66% |
| 80 | 789 | 0.29% |
| 60 | 148 | 0.054% |
| 40 | 28 | 0.010% |
| 21 | 6 | 0.002% |

總權重 ≈ 275418 (n=21..120 each appearing once)

---

## 目前狀態 (Current Status)

### 最佳已驗證版本：Total Score = **3.2708** (新評分)

主要參數：
- `TIME_LIMIT = 8.00 秒`
- `W_VIOL = h0 × 7.5` (新評分 sweet spot；舊評分 sweet spot 為 h0×9)
- `W_AREA = h0/a0` clamped [0.01, 2.0]
- `W_BOUNDARY = 10.0` ← **注意：程式碼當前為 100.0，但未編譯/驗證**
- `MAX_PACK_SIZE = 120`

### ⚠️ 程式碼 vs 已驗證狀態差異
> `optimizer_claude.cpp` 目前 `W_BOUNDARY = 100.0`（未驗證的實驗）。
> 若要復現最佳分數 3.2708，請先將 `W_BOUNDARY` 改回 `10.0`，重新編譯。

### 關鍵組件
1. **初始 permutation**：connectivity-driven order，依優先序重排
   - 0: cluster blocks（按 gid 連續）
   - 1: LEFT/BOTTOM boundary blocks
   - 2: 一般 blocks
   - 3: RIGHT/TOP boundary blocks

2. **Skyline decode**：
   - `pack_cluster_multirow`: ceil(sqrt(nm)) 行寬
   - `pack_cluster_anchored`: 函式存在但已不被呼叫（歷史失敗）

3. **W_VIOL/W_AREA 動態校正**：
   ```cpp
   W_AREA = clamp(h0/a0, 0.01, 2.0);
   W_VIOL = max(50.0, h0 * 7.5);
   ```

4. **SA 主迴圈 (8 秒, T 200 → 0.05)** — 8 種 move 類型
   - 0.30 swap / 0.14 relocate / 0.08 connectivity / 0.10 resize
   - 0.08 rotate / 0.12 MIB unify / 0.08 cluster adjacency / fallback swap

5. **Post-processing 序列**（總 ~0.2s）：
   1. `cluster_snap`：disconnected cluster member 朝 anchor centroid 滑動，**guarded by proxy_cost** (20 passes, dominant axis with fallback)
   2. `cluster_boundary_snap`：cluster 整體向 boundary edge rigid translation，**guarded by proxy_cost**
   3. `boundary_snap`：non-cluster boundary block 推向 required edge
   4. `slack_hpwl_opt`：non-cluster non-fixed block 沿 HPWL force 滑動（boundary 方向受限），budget 0.15s
   5. `boundary_snap` 再次：修正 hill-climber 偏移

### Violation breakdown（觀察結果, top cases）
| Case | n | cost | vBd | vCl | vMb |
|------|---|------|-----|-----|-----|
| 99 | 120 | 3.89 | **22** | 2 | 0 |
| 98 | 119 | 3.92 | 17 | 3 | 0 |
| 97 | 118 | 3.56 | 15 | 4 | 1 |
| 95 | 116 | 3.86 | 20 | 1 | 1 |
| 91 | 112 | 3.94 | 19 | 3 | 1 |

**Boundary violations 主導**（12-22 per top case）；cluster ~3-5；MIB ~0-1。

---

## 分數歷程

### 舊評分公式
| 版本 | 分數 |
|------|------|
| baseline (MAX_PACK=12, 1.45s SA) | 6.484 |
| + 8s SA + dynamic W_VIOL ×1.0 | 4.5501 |
| + boundary_snap | 4.2173 |
| + cluster_snap (viol guard) | 3.9557 |
| + MAX_PACK_SIZE=120 | 3.9481 |
| + cluster_snap 20 pass + axis retry | 3.9467 |
| + W_VIOL ×1.5 (h0×9) | **3.7944** |
| TIME_LIMIT=12s | 4.3120（退步） |

### 新評分公式
| 版本 | 分數 |
|------|------|
| 舊最佳 (W_VIOL ×1.5) | 3.4226 |
| W_VIOL ×0.75 (h0×4.5) | 3.4458 |
| W_VIOL ×1.0 (h0×6.0) | 3.3715 |
| **W_VIOL ×1.25 (h0×7.5)** | **3.3259** |
| W_VIOL ×1.375 (h0×8.25) | 3.4454 |
| nearest anchor cluster_snap | 3.5132 |
| slack_hpwl_opt slack=1.0 | 3.3303 (neutral) |
| T_END = 0.01 | 3.3779 |
| cluster move 0.08→0.16 | 3.4695 |
| cluster_hpwl_opt rigid HPWL translation | 3.3548（略退） |
| + cluster_boundary_snap (viol guard) | 3.3762 |
| + cluster_boundary_snap (**proxy_cost guard**) | 3.3183 |
| + cluster_snap **(proxy_cost guard)** | **3.2708** ← current best |

---

## 這個階段想解決的問題

### A. Boundary violations 仍是主要瓶頸（最高 leverage）
- Top case 仍有 11-22 個 boundary violations
- 這些 block 無法被 `boundary_snap` 移動，原因是 **slack = 0**（周圍 block 阻擋）
- `cluster_boundary_snap` 只能處理 cluster 內的 boundary blocks
- 推測剩餘違反主要是 **非 cluster 的 boundary block 被非 boundary block 卡住**

### B. HPWL gap 在 top cases 仍偏高
- 即使 SA 8s 還是無法收斂到接近 best HPWL
- post-processing 受 slack 限制，無法做大幅修正

### C. Area gap 也偏高（0.5-0.8）
- bbox 中沒有任何機制把內部 block 拉向中心（壓縮 bbox）

### D. 未測試的實驗（待跑）
- `W_BOUNDARY = 100`（程式碼已改，未編譯）
- W_BOUNDARY 提升可能讓 SA 的 boundary force 梯度更明顯

---

## 預期目標

### 短期（1–3 個迭代）
- **目標 1**：boundary violations 中位數從 17 降到 ≤ 10
  - 需要找出「slack=0 的 boundary block」要如何處理
- **目標 2**：Total Score < 3.20（從 3.2708 再降 2-3%）
- **目標 3**：完成 `W_BOUNDARY = 100` 實驗，驗證或回退

### 中期（4–6 個迭代）
- **目標 4**：實作「boundary-aware swap」或「chain push」處理 slack=0 案例
- **目標 5**：壓縮 bbox 來降低 area_gap（bbox shrinking post-processing）
- **目標 6**：Total Score < 3.10

### 長期
- **目標 7**：替換 BL packer 為 Sequence Pair / B*-Tree
- **目標 8**：把 violation handling 從「penalty」改為「constraint repair after each move」

---

## 未來發展方向

### 1. 解決 slack=0 boundary 違反
- **Chain push**: 若 boundary block A 想往 LEFT 但被 B 擋住，嘗試把 B 也往 LEFT 推，再推 A
- **Swap**: 若 A (B_LEFT) 在 x=2，B (no constraint) 在 x=0，swap 之
- **整列重排**: 把 boundary-required block 強制塞入 leftmost 列

### 2. SA 中強化 boundary handling
- 已嘗試：W_BOUNDARY 從 10 → 100（待驗證）
- 可試：W_BOUNDARY 隨 SA 進度 ramp up（早期 10、後期 200）
- 可試：boundary 違反在 calc_violation 中改為 squared/exp 函數，懲罰大違反

### 3. Bbox shrinking
- 後處理：找 bbox 邊緣的非 boundary block，往內推（slack-guided）
- 重新計算 bbox，迭代

### 4. SA 演進
- **Restart with perturbation**: 偵測 stagnation 時對 best_pos 局部擾動
- **Move 機率自適應**: acceptance rate-driven adjustment
- 已試過但失敗：multi-restart（兩段 4s 不足）、cluster move 加倍

### 5. 結構化 violation 修復
- 利用 `viol_breakdown.py`（已建立）監控 vBd/vCl/vMb 變化
- 對 cluster 違反：BFS 找最大 component，剩餘 component 做 group 移動（已部分實作於 cluster_snap）
- 對 boundary 違反：考慮 ROW-level 重排，把 LEFT 需求的 block 整列移到 leftmost

### 6. 效能優化（若 TIME_LIMIT 需提升）
- `calc_violation` 目前 O(N² + Σ cluster_size²)，可加 spatial index
- `skyline_decode` 每個 SA move 重新呼叫，可改 incremental
- post-processing 的 `proxy_cost` guard 可只重算受影響子集
- 當前每個 case 8s 中，post-processing 約 ~0.5s（含 calc_violation 多次呼叫）

### 7. Cluster preplaced 改進
- `pack_cluster_anchored` 函式存在但已不被呼叫（歷史測試擴大 area）
- 在新評分下可重新評估：cluster anchored 在 area_gap 較不重要時可能有用

---

## 環境

- **主程式**: `optimizer_claude.cpp` (C++) + `optimizer_claude.py` (Python wrapper)
- **Conda env**: `C:\Users\Nordra\.conda\envs\iccadv\python.exe`
- **Compiler**: `C:\msys64\ucrt64\bin\g++.exe`

### 編譯
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet"
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o optimizer_claude.exe optimizer_claude.cpp 2>&1
```

### 評估（13-15 分鐘）
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_claude.py 2>&1 | Select-Object -Last 12
```

### 單 case 測試
```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_claude.py --test-id 99
```

### 分析
```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_results.py
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" viol_breakdown.py    # 已修好，顯示 vBd/vCl/vMb
```

---

## 已知 Bug / 注意事項

- **PowerShell 用分號或 `if ($?) {...}` 連接指令，不能用 `&&`**
- **評估需要 13–15 分鐘（100 個 case × 8 秒）**
- **`optimizer_claude.cpp` 中 `W_BOUNDARY` 目前為 100.0（未驗證）**，
  最佳分數對應 `W_BOUNDARY = 10.0`
- **`analyze_violations.py` 無法執行**（lite_dataset_test 缺失）；
  改用 `viol_breakdown.py`（已建立）
- **`viol_breakdown.py` 的 hpwl_gap/area_gap 為 -0.9～-1.0 範圍**，
  這是因為使用 `metrics[0]`/`metrics[1]` 當 baseline，
  但官方 baseline 不同；數字不可直接對比
- **`pack_cluster_anchored` 函式仍在程式碼但已不被呼叫**

---

## 檔案結構

```
FloorSet/
├── optimizer_claude.cpp    ← 主程式 (C++)
├── optimizer_claude.py     ← Python wrapper
├── optimizer_claude.exe    ← 編譯輸出
├── CLAUDE.md               ← 本檔案
└── iccad2026contest/
    ├── iccad2026_evaluate.py       ← 新版評估腳本
    ├── analyze_results.py          ← top cases 顯示（可用）
    ├── viol_breakdown.py           ← violation 分項（已建立，可用）
    ├── analyze_violations.py       ← 舊版，無法跑
    ├── check_viols.py              ← 舊版，無法跑
    └── optimizer_claude_results.json  ← 最新評估結果
```

---

## 給下一個 session 的優先建議

1. **第一件事**：把 `W_BOUNDARY` 從 100 改回 10（或測試 100 是否真的有幫助）
2. **第二件事**：跑 `viol_breakdown.py` 確認 vBd 仍是主導
3. **第三件事**：實作「slack=0 boundary block」的處理機制
   （chain push 或 swap-based snap）
4. **第四件事**：如果 boundary 已壓低，下一個方向是 bbox shrinking
   來降低 area_gap
