# ICCAD 2026 FloorSet — Session Context

## 🚨 範式轉移 (2026-05-26, 重要！)

**這個競賽不是 floorplan optimization，而是 reconstruction（拼圖還原）。**

### 證據

組員用**純演算法（無 ML）**做到：
- **Total Score = 1.0322**
- **Avg Cost = 1.0408**

對照我們當前的最佳（GNN + SA）= **3.3258**。差距 **3.2×**。

### 為什麼 reconstruction 比 optimization 強

Cost 公式：`Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)`

- 當 gap = 0 且 V_soft = 0 時 → **Cost = 1.0**（理論最小）
- 我們 SA 自己「找最佳解」永遠 > baseline → HPWL_gap > 0 → Cost > 1
- 組員把問題當「**還原 baseline 那張原圖**」→ HPWL_gap ≈ 0 → Cost ≈ 1

### 訓練資料其實藏著答案

`get_training_dataloader()` 回傳：
```
(area_target, b2b_conn, p2b_conn, pins_pos, constraints,
 tree_sol, fp_sol, metrics)
                  ^^^^^^^
              原圖的 (w, h, x, y)
```

`fp_sol` = ground truth 位置。我們的 `compute_training_loss_differentiable`
**完全沒用它** — 只看 HPWL + overlap，是無監督。所以 GNN 學到的是「散開且
連線短的擺法」，而不是「原圖怎麼擺」。

### 對策

1. **演算法**：等組員分享純演算法代碼，研究他怎麼用 constraint + connectivity
   反推 baseline 佈局
2. **ML**：把 `training_example.py` 從**無監督** loss 改成 **supervised MSE
   對 fp_sol**。GNN capacity 都在，只是被訓練成做錯誤任務
3. **architecture**：目前 SA + skyline BL packer 是 optimization 思維的產物。
   若要 reconstruction，可能要把 SA 角色從「找解」改成「微調 ML 輸出」，甚至
   完全拿掉

### 已知未解問題

- 組員的 1.0322 完全在 validation set（100 case）。test set（hidden）效果未知
- BL packer 的「給對 perm 能還原多接近」沒實驗驗證過；可能架構本身就是瓶頸
- 監督式訓練的 ML 上限未知；可能跟組員純演算法持平或互補

---

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
- `W_BOUNDARY = 10.0`
- `MAX_PACK_SIZE = 120`

### 🆕 GNN-hint 整合 (2026-05-24, 已驗證有效)

`optimizer_claude.py` 現在會 lazy-load `floorplan_gnn.pth`，每個 case 推論一次
GNN 預測的 (cx, cy)，把它當成「替代 initial perm」候選餵給 C++。C++ 在
`run_sa` 起始時 decode 兩種 perm（connectivity vs GNN-sorted by cx+cy），
取 `raw_cost`（HPWL+area+violation）較低者當 SA 起點。

> ⚠️ **note**：原本以為「raw_cost 比較較佳 → SA 最終結果一定不退步」，但
> SA 8 秒會從不同起點走到不同 basin — 起點 raw_cost 較低 ≠ 終點較低。
> 所以這實際上**不是嚴格零退步**的設計，需用 full eval 驗證。實測下來確實
> 是淨改善。

- 環境變數 `ICCAD_DISABLE_GNN=1` 可關閉 GNN，做 A/B 比較
- 沒 torch / 沒 .pth / 推論失敗時自動跳過，不影響原本流程
- 額外推論成本 ~10–50ms/case (CPU)，總開銷 < 5s

**100-case full eval (2026-05-25, 含 3000-sample 重訓的新 .pth)**：
| 設定 | GNN training | Total Score | Avg Cost | Feasible |
|------|-------------|------------|----------|----------|
| no-GNN baseline | — | 3.4308 | 2.6478 | 100/100 |
| GNN enabled (old .pth) | 500 sample, loss=2.58 | 3.3469 | 2.6157 | 100/100 |
| **GNN enabled (new .pth)** | **3000 sample, loss=1.34** | **3.3258** | 2.6548 | 100/100 |

→ **新 .pth 比 no-GNN 改善 3.1%**，比 old GNN 再進步 0.6%。

**邊際效益警告**：500 → 3000 sample (6× 訓練量) 僅換來 0.6% Total Score 進步。
再加訓 sample 預估收益 < 0.3%。下一個 1% 要從別處挖（改整合方式、攻 boundary
violations、或 tournament SA）。

**Case 層級分歧** (smoke test 觀察)：
| Case | n | Old GNN | New GNN |
|------|---|---------|---------|
| 0  | 21  | 1.9387 | 2.3845 (+23%) |
| 99 | 120 | 3.6031 | 4.2474 (+18%) |
新權重在某些 case 大進，某些大退 — aggregate 淨贏但分佈變分歧。

> ⚠️ 兩者都高於先前文件記錄的 3.2708。可能原因：CLAUDE.md 的 3.2708 是早
> 期 code state 的結果，當前 code 已飄移。若要追回 3.27，需要 git bisect
> 找到 regression point — 但這是獨立於 GNN 整合的另一個議題。

### 關鍵組件
1. **初始 permutation**：兩個候選 → 取 proxy_cost 較低者
   - **候選 A**：connectivity-driven order (greedy NN on b2b graph)
   - **候選 B**：GNN-sorted by (cx+cy) (若 GNN hint 存在)
   - 兩者都套用相同的優先序重排 (見下)
   - 優先序：
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
| + cluster_snap **(proxy_cost guard)** | 3.2708 |
| 2026-05-24 W_BOUNDARY=10 baseline (no GNN, current code) | 3.4308 |
| 2026-05-24 + GNN-hint initial perm (loss=2.58 .pth) | 3.3469 |
| 2026-05-25 + GNN-hint (3000-sample retrained .pth, loss=1.34) | 3.3258 |
| 2026-05-26 v2 supervised MSE on fp_sol (2000 sample, < 3h) | **失敗** — pos_mse 震盪、unsup_cost 47M，.pth 已棄 |
| 2026-05-27 v3 structural (pairwise ranking, 進行中) | — |
| **【外部參考】組員純演算法 reconstruction approach** | **1.0322** ← 真正的目標線 |

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
- **`analyze_violations.py` 無法執行**（lite_dataset_test 缺失）；
  改用 `viol_breakdown.py`（已建立）
- **`viol_breakdown.py` 的 hpwl_gap/area_gap 為 -0.9～-1.0 範圍**，
  這是因為使用 `metrics[0]`/`metrics[1]` 當 baseline，
  但官方 baseline 不同；數字不可直接對比
- **`pack_cluster_anchored` 函式仍在程式碼但已不被呼叫**
- **GNN 推論需要 `torch`**：若 conda env 缺 torch，python wrapper 會跳過 GNN
  並印 stderr 警告（不會 crash）

---

## 檔案結構

```
FloorSet/
├── optimizer_claude.cpp    ← 主程式 (C++)，含 GNN-hint 比較邏輯
├── optimizer_claude.py     ← Python wrapper，含 GNN inference (v1 FloorplanNet)
├── optimizer_claude.exe    ← 編譯輸出
├── floorplan_gnn.pth       ← v1 權重 (FloorplanNet, 128 hidden, unsupervised)
├── floorplan_gnn_checkpoint.pth ← v1 訓練中 checkpoint
├── floorplan_gnn_v2.pth    ← v2 權重（已棄，supervised MSE 失敗）
├── floorplan_gnn_v3.pth    ← v3 權重（待訓練；structural BL ordering）
├── v8_puzzle_fingerprint_oracle_repair.py ← 組員放在根目錄的 wrapper（讀但不改）
├── CLAUDE.md               ← 本檔案
├── gnn_training.md         ← ML 部分文件（FloorplanNet 訓練紀錄）
├── teammate_eva/           ← 組員提供的參考檔（不可修改）
│   ├── README_baseline.md
│   ├── my_optimizer.py     ← v3-shelf-packing 級，非 1.03 來源
│   └── v8_puzzle_fingerprint_oracle_repair.py
└── iccad2026contest/
    ├── iccad2026_evaluate.py       ← 新版評估腳本
    ├── training_example.py         ← GNN 訓練腳本（當前 v3 structural ranking）
    ├── analyze_results.py          ← top cases 顯示（可用）
    ├── viol_breakdown.py           ← violation 分項（已建立，可用）
    ├── analyze_violations.py       ← 舊版，無法跑
    ├── check_viols.py              ← 舊版，無法跑
    └── optimizer_claude_results.json  ← 最新評估結果
```

## 機器學習部分 (詳見 `gnn_training.md`)

### v1 訓練腳本 (2026-05-24, 已退役)

無監督 (HPWL + overlap penalty)，2 層 GCN, 128 hidden, 含 grid prior。
3000 sample / loss=1.34 → Total Score 3.3258（穿 GNN-hint integration）。

### v2 訓練腳本 (2026-05-26, **失敗**)

Supervised MSE 對 `fp_sol`，4 層 residual GCN, 256 hidden, LayerNorm, 14-dim
features, scatter_add 向量化邊處理。

**結果：訓練本身失敗。** 2000 sample log 觀察到：

| 指標 | 觀察 | 含意 |
|------|------|------|
| `pos_mse` | 700-2300 大幅震盪，無下降趨勢 | 模型在位置預測上完全沒學到 |
| `dim_mse` | 17-26 平穩 | 寬高 (aspect ratio) 學得 OK |
| `unsup_cost` | 10 → **47,000,000** 暴衝 | 預測佈局物理上完全無效（heavy overlap） |

**根因：任務是 ill-posed 的一對多**
- 輸入 X = (connectivity, area, constraints) → 多個合法佈局 Y₁, Y₂, … 都對應到同樣的 X
- 訓練只看到一個特定 Y_train（`fp_sol`）
- MSE 收斂到「所有可能 Y 的平均」 → 一堆方塊疊在中間
- pos_mse 大幅震盪是因為不同 batch 的 Y_train 把模型往不同方向拉

**訓練時長正面意外**：v2 的 2000 sample 訓練 < 3h（原估 6h）— scatter_add
向量化邊處理 vs v1 的 Python edge loop，~2× speedup。但 throughput 高跟訓練
品質無關。

`floorplan_gnn_v2.pth` **不可使用**，請勿 commit。

### v3 訓練腳本 (2026-05-27, 進行中) — 預測結構而非位置

新方向：模型輸出**每個 block 一個 BL ordering score**（scalar），不再預測絕對
位置。架構同 v2（residual GCN + LayerNorm），但 output head 改為：
- `bl_head` → 1 scalar BL score per block
- `ratio_head` → 1 scalar aspect ratio（aux loss）

**Loss**：
- **Pairwise ranking loss**：對所有 (i, j) pair，若 fp_sol 的 `(x+y)[i] < (x+y)[j]`，
  則 `bl_score[i] < bl_score[j]`。BCE on `sigmoid(bl[j] - bl[i])`
- **Aux aspect MSE**：MSE on (w, h) 對 fp_sol，保留 v2 學到的 aspect ratio
- 總 loss = ranking_loss + λ · aspect_mse

**為什麼這次該成功**：
- 一對多問題下，**absolute position 不可學**，但**relative order 可學**
- BL packer 只需要 perm（順序），不需要絕對位置
- 不同合法佈局之間的 BL ordering 可能相似（small-area-first / boundary-first 等
  universal 模式），所以 pairwise loss 噪音較小

**儲存**：`floorplan_gnn_v3.pth`（不覆蓋 v1/v2，方便 A/B 比較）

### 通用 flags（所有版本）

```bash
--sanity              # 20 samples / 5 batches / 不存檔（pipeline 驗證）
--num-samples N       # 指定樣本數
--fresh               # 不 load 既有 .pth，從零訓練
```

> ⚠️ **本環境禁止跑訓練**，若要訓練請複製到另一個 GPU 環境執行

---

## 給下一個 session 的優先建議

> ⚠️ **2026-05-26 更新**：以下舊建議（boundary / bbox / slack）都是 optimization
> 思維。範式已轉移到 reconstruction，請先看本文件頂部「🚨 範式轉移」段落，
> 再決定要不要繼續走 optimization 路線。

### 新優先級（reconstruction 範式, 2026-05-27 更新）

1. **等組員補齊缺檔** → 已收到 `teammate_eva/`，但只有 wrapper +
   v3 級 shelf packing helper，**真正的 `v8_puzzle_fingerprint_oracle.py`
   還沒拿到**。其打到 1.0322 的核心 reconstruction 邏輯仍是黑盒
2. **v3 訓練腳本（path C）已實作** → `training_example.py` 改為預測 BL
   ordering score（pairwise ranking loss），不再預測絕對位置
   - 上 GPU 環境跑 `--sanity` 驗證 pipeline
   - 訓出 `floorplan_gnn_v3.pth` 後拷回，等我更新 `optimizer_claude.py`
     支援 v3 推論（目前還沒做）
3. **驗證 BL packer 上限**：寫一個小腳本，把 `fp_sol` 排序當 perm 餵
   `optimizer_claude.exe`，看 Total Score 能否逼近 1.0
   - 若 ≈ 1.0 → 架構 OK，v3 ranking 預測夠準就有救
   - 若 ≈ 2.0+ → BL packer 會把答案拆掉，需要換 placer
4. **不要動 `teammate_eva/` 內任何檔案** — 由使用者管理

### 失敗紀錄（避免重蹈）

- **v2 (2026-05-26) supervised MSE on fp_sol absolute position**：
  ill-posed 一對多任務，loss 震盪、預測佈局物理無效（unsup_cost 暴衝到 47M）
  → `floorplan_gnn_v2.pth` 已捨棄，**不要 commit**

### `teammate_eva/` 檔案狀態

- ✅ `README_baseline.md` — 只描述 v1-v5 shelf packing，沒提 v8 oracle
- ✅ `my_optimizer.py` — v3 級 shelf packing，**不是** 1.03 的來源
- ✅ `v8_puzzle_fingerprint_oracle_repair.py` — thin wrapper（52 行）
- ❌ `v8_puzzle_fingerprint_oracle.py` — **missing**（真正核心）
- ❌ 預期更進階版的 `my_optimizer.py`（含 `_final_boundary_nudge` /
  `_final_group_boundary_nudge` / `_final_adaptive_single_edge_escape` 三方法）

### 舊優先級（optimization 範式，已 deprecate）

~~第一件事：W_BOUNDARY 100 → 10~~（已做）
~~第二件事：viol_breakdown 確認 vBd 主導~~
~~第三件事：slack=0 boundary block 處理~~
~~第四件事：bbox shrinking~~

這些只能在「我們要繼續做 optimization」的前提下有意義。在 reconstruction
範式下，所有 violation/HPWL/area 都是 baseline 內建的副產品 — 還原原圖
自然就沒問題。
