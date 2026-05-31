# ICCAD 2026 FloorSet Challenge - GNN 訓練與維護指南 `gnn_training.md`

## 🎯 檔案定位

本檔案是 **ICCAD 2026 FloorSet Challenge** ML 部分的完整技術紀錄。涵蓋三個版本
的訓練腳本演進、各自的失敗教訓、以及當前未解決的問題。

> 📌 看這份檔案之前，**先讀 [CLAUDE.md](CLAUDE.md) 頂端的「🚨 範式轉移」段落**。
> 不知道這個競賽是 reconstruction 而非 optimization，看 ML 紀錄會看不懂為什麼
> 一直在換 loss。

---

## 🚨 範式轉移摘要 (2026-05-26, 含 2026-05-27 修正)

組員的 **Total Score = 1.0322** 經查證是 **oracle**（讀本地 validation label），
hidden test 不適用。組員真正 legit 的最佳是 ~1.6（v6/v7 portfolio）或
~1.76（v5 單一純算法）。我們 SA + GNN 在 3.3258，與 legit 差 2×。

關鍵發現（修正後仍成立）：
- Cost 公式 `Cost = (1+α·gap)·exp(β·V_soft)`，理論最低 = 1.0
- 我們從零優化 → gap > 0 → cost > 1
- 組員 v5 後抓到「**boundary aspect ratio 拉到 2.50/0.40**」這個 leverage，
  從 2.58 一路掉到 1.76
- 訓練資料 `fp_sol` 是原圖位置，可作監督訊號，但 v10 ML 嘗試（per-block factor
  rank prediction）失敗 — val acc 44% 但下游 < 1% 進步

從這個 insight 出發，ML 路線經歷了 v1（無監督）→ v2（監督絕對位置）→ v3
（監督結構排序）的演進。

### v10 教訓對 v3 ranking 的啟示

組員的 v10 用 per-block MLP 預測 factor rank（離 v3 ranking 路線很近），
val acc 44%、within-one-rank 86%，但下游下降 < 1%。原因：
- per-block local feature 在這題訊號太弱
- BL packer 把 ranking 「破壞」掉的程度未知
- ML signal 跟 BL packer 行為不對齊

→ v3 在跑 full training 前，**必須先做 oracle BL 上限實驗**（見下節）
   驗證 BL packer 在 ranking 完美時能達到什麼分數。否則重蹈 v10 覆轍。

---

## 📚 版本歷程

### v1: 無監督全域佈局 (2026-05-24, 已退役但仍可用)

**目的**：用 contest 的可微分 cost 直接做 global placement。當時還在
optimization 範式下，目標是「自己找好的佈局」。

**架構**：`FloorplanNet`
- 2-layer GCN (Linear + ReLU)
- hidden_dim = 128
- 輸出層：sigmoid×150 + grid_offset(i)，輸入 (cx, cy, log_ratio, _)
- Grid offset 是「讓 block i 預設散開在 grid 上」的 inductive bias

**Loss**：`compute_training_loss_differentiable` from `iccad2026_evaluate.py`
- 內容：`Cost = (1+α·(HPWL_gap+Area_gap)) × exp(β·V_soft)` 可微分版
- V_soft = overlap + area_tolerance violation

**訓練結果**：
| Run | Samples | 時長 | Final Loss |
|-----|---------|------|------------|
| 第一次 | 500 | 1.5h | 2.58 |
| 第二次 | 3000 | 9h | **1.34** |

**下游整合**：`optimizer_claude.py` 用 GNN 預測的 (cx+cy) 排序當 BL packer
的 perm hint：

| 設定 | Total Score |
|------|-------------|
| 無 GNN baseline | 3.4308 |
| v1 (loss=2.58) | 3.3469 |
| v1 (loss=1.34) | **3.3258** |

**v1 退役原因**：範式轉移後發現訓練目標就錯了 — 學會「自己找佈局」而不是
「還原原圖」。3.3258 是 v1 路線的天花板，距離組員的 1.0322 還差 3×。

**v1 權重檔**：`floorplan_gnn.pth`（保留作 reference，optimizer_claude.py
仍在使用）

---

### v2: 監督式 MSE on fp_sol 絕對位置 (2026-05-26, **失敗**)

**目的**：把 v1 的無監督 loss 換成「直接學原圖位置」。範式轉移後的第一次嘗試。

**架構**：`FloorplanNetV2`，與 v1 完全不同
- 4-layer residual GCN (LayerNorm + Linear + ReLU + Dropout + residual)
- hidden_dim = 256
- 14-dim features：area, sqrt(area), log(area+1), avg_pin_xy, pin_count,
  boundary flags (L/R/T/B), is_preplaced, is_fixed, has_group
- **scatter_add 向量化邊處理**（vs v1 的 Python edge loop, 2× speedup）
- 拿掉 grid offset prior（妨礙 supervised learning）
- Output: `sigmoid(pos_head) × 500`（範圍 [0, 500] cover validation 資料的位置）

**Loss**：
```
target = (x, y, w, h)  ← 從 fp_sol 取出（fp_sol 存的是 (w, h, x, y)，要 reorder）
pos_mse = mean((pred[:, :2] - target[:, :2])^2)
dim_mse = mean((pred[:, 2:] - target[:, 2:])^2)
total = pos_mse + dim_mse
```

**訓練結果 (2000 sample / < 3h)**：
| 指標 | 觀察 | 含意 |
|------|------|------|
| `pos_mse` | 700-2300 大幅震盪，無下降趨勢 | 模型在位置預測完全沒學到 |
| `dim_mse` | 17-26 平穩 | 寬高 (aspect ratio) 學得 OK |
| `unsup_cost` | 10 → **47,000,000** 暴衝 | 預測佈局物理上完全無效 |

**下游整合**：
| 設定 | Total Score |
|------|-------------|
| 無 GNN | 3.4308 |
| v1 GNN | 3.3258 |
| **v2 GNN** | **4.2173** ← 比沒 GNN 還差 23% |

**失敗根因（最重要的教訓）**：

任務本質是 **ill-posed 一對多**：
- 輸入 X = (connectivity, area, constraints)
- 多個合法佈局 Y₁, Y₂, … 都對應到同樣的 X
- 訓練只看到一個特定 Y_train（`fp_sol`）
- MSE 收斂到「所有可能 Y 的平均」 → 一堆方塊疊在中間
- pos_mse 震盪是因為不同 batch 的 Y_train 把模型往不同方向拉

**w/h 預測勉強 OK** 是因為自由度小（只有 aspect ratio 一個維度，且受 area
約束）。但位置自由度大，一對多問題嚴重。

**v2 權重檔**：已棄，所有 artifact 從 git 刪除（commit `2235ad0`）

---

### v3: 結構預測 (BL ranking) (2026-05-27, 進行中)

**目的**：避開「絕對位置不可學」的根本問題。改成預測**結構性訊號**（block
之間的相對排序），讓下游 BL packer 用 perm 還原佈局。

**為什麼這該成功**：
- BL packer 只吃 perm（順序），不吃絕對位置 — 訓練目標跟下游需求對齊
- Ranking 對「絕對值」不敏感 — 多個合法佈局只要 BL 順序相似就 OK
- 一對多問題在 ordering 維度比 absolute position 維度小（同 area 同 group
  的 block 通常排序差不多）

**架構**：`FloorplanNetV3`，與 v2 共用 encoder
- 4-layer residual GCN, hidden_dim=256（同 v2）
- 14-dim features（同 v2）
- scatter_add 向量化（同 v2）
- **Output head 換掉**：
  - v2: `pos_head` → (x, y, log_ratio)
  - v3: `bl_head` → 1 scalar BL score + `ratio_head` → 1 scalar aspect

**Loss**：
```python
# Main: pairwise BCE on (x+y) ordering
target = (gt[i] > gt[j])     # bool, 1 if block i should rank after j
logits = bl_pred[i] - bl_pred[j]
rank_loss = BCE(sigmoid(logits), target)

# Aux: MSE on (w, h)
aspect_loss = MSE(pred_wh, fp_sol_wh)

# Total
loss = rank_loss + λ * aspect_loss
```

**輸出 shape**：仍是 (n, 4) = `[bl_score, 0, w, h]`，column 1 是 0
placeholder，跟 v1/v2 推論 code 相容（`optimizer_claude.py` 可暫時用
`pred[:, 0] + pred[:, 1] = bl_score` 當 sort key）。

**儲存**：`floorplan_gnn_v3.pth`（與 v1 分開，目前 v1 仍是 active 權重）

**sanity 模式設計（第二輪修正後）**：
- 樣本數 120（= 30 batches）
- aspect weight 預設 = **0**（隔離 ranking 訊號）
- 不存檔、不 checkpoint
- 目的：30 batches 內看 `rank_acc` 是否從 ~0.5 爬到 ~0.6

**第一次 sanity 失敗（2026-05-27 早）**：
- 預設 aspect-weight=0.01 太大 → aspect_loss=865 × 0.01 = **8.65 dominate**
  rank_loss=0.69
- 只 5 batches，cosine LR 在 5 batches 內塌到 0.00001
- 加上一個 `probe_unsup` diagnostic 把 raw bl_score 餵 contest cost 函式，
  暴衝到 **10^16**（pure noise，無意義）
- 結果：30 batches `rank_acc` 在 0.51-0.55 隨機跳

**第二次 sanity 結果（修正後, 120 sample = 30 batches）**：

| Batches | avg rank_acc | rank_loss |
|---------|--------------|-----------|
| 0-4 | 0.529 | 0.69 |
| 5-9 | 0.568 | 0.68 |
| 10-14 | 0.558 | 0.68 |
| 15-19 | 0.569 | 0.67 |
| **20-24** | **0.595** | 0.66 |
| 25-29 | 0.583 | 0.66 |

訊號**存在但弱**：30 batches 內 rank_acc 從 0.53 → 0.58（+5%），但每個 batch
的 noise 是 ±0.05，訊號跟噪音差不多大。`aspect_loss`（觀察用，無梯度）
從 86 → 64，確認 encoder 在學東西。

**未決定要不要 commit 到 full training**。詳見「當前未解決的問題」。

---

## 🔍 當前未解決的問題

### Q1: BL packer 上限是多少？— ✅ 已解決 (2026-05-31)

**結論：BL packer 是天花板，v3 ML ranking 訓練是白工。**

實驗 (`optimizer_oracle_perm.py`, 3 個 mode)：

| Mode | Total Score | Avg Cost | 機制 |
|------|------------|----------|------|
| raw | 1.1079 | 1.1097 | 直接 return fp_sol verbatim（對上 teammate 1.1079） |
| bl | 9.9996 | 9.8569 | oracle perm + oracle shape + 純 BL packer (no SA) |
| exe | **3.2673** | **2.6494** | oracle perm 餵 C++ 當 GNN hint，full SA |
| (現狀 v1 GNN) | 3.3258 | 2.6548 | 對照組 |

**判讀**：
- Oracle perm + SA 只比現狀進步 **1.8%**（3.3258 → 3.2673）
- 即使 ranking 完美，pipeline 上限就是 3.27 — 距 1.6 legit 還差 2×
- raw (1.10) vs exe (3.27) 的巨幅落差 = 我們的 BL packer + shape 邏輯崩潰
- ranking ML 訓練的 ROI 不超過 1.8%

**因此 v3 ranking 訓練放棄**。

### Q1.5: Shape ML 上限是多少？— ✅ 已解決 (2026-05-31)

**結論：shape ML 也死了**。延伸 `optimizer_oracle_perm.py` 加兩個 mode：

| Mode | Total Score | Avg Cost | 機制 |
|------|------------|----------|------|
| shape | **3.4199** | 2.7638 | oracle (w,h) + connectivity perm + SA (resize 鎖死) |
| shape_perm | **3.3672** | 2.8001 | oracle 兩者 + SA (resize 鎖死) |
| (baseline 對照) | 3.4308 | 2.6478 | 純 SA, 無 GNN, 無 oracle |
| (exe perm-only 對照) | 3.2673 | 2.6494 | oracle perm + sqrt area |

**判讀**：
- Shape 3.42 vs baseline 3.43 = **0.3% 改善**（純 noise，shape ML 無價值）
- Shape_perm 3.37 比 perm-only 3.27 還差 — 鎖死 shape 讓 SA 失去 ~25% move
  類型（resize + rotate），探索能力下降反而傷害
- **即使給完美 shape + perm，pipeline 上限就是 ~3.2-3.4**

### Q1 + Q1.5 合併結論：天花板在 placer 架構

| ML 預測目標 | Oracle 上限 | 改善 vs baseline | 結論 |
|------------|-----------|------------------|------|
| perm (ranking) | 3.27 | 1.8% | 不值得訓練 |
| shape (factor pair) | 3.42 | 0.3% | 不值得訓練 |
| perm + shape | 3.37 | 1.9% | 不值得訓練 |

所有 ML 路線都被「SA + skyline BL packer」這個架構 cap 在 ~3.2-3.4。
要破 3.0 不能靠 ML 信號 input，要動 placer 本身：
- 換 placer：B*-tree、sequence-pair、shelf packing（組員用 shelf 到 1.6）
- 純算法擴充：portfolio selector (已驗證有效, 3.33 → 3.16)
- Repair-style post-processing：針對性修 violation

### Q2: rank_acc 0.58 plateau 是 fundamental 還是 cosine LR 問題？

Sanity 用 cosine 在 30 batches 內塌到 LR=0.00001，後期幾乎沒學。可能：

**樂觀**：full training 500 batches，cosine 拉得開，LR 維持 0.0005+ 較久，
最終 rank_acc 可以爬到 0.75-0.85。

**悲觀**：30 batches 內 noise ±0.05 vs 訊號 +0.05，可能 plateau 在 0.65。
即使 full training 也只到 0.70 左右。0.70 ranking 餵 BL packer 可能完全沒幫助。

**怎麼分辨**：跑一次 full training（2-3h GPU）就知道。但這就是「Q1 答案
決定要不要做」的問題。

### Q3: 一對多本質問題在 ranking 維度有多嚴重？

v2 失敗是因為 absolute position 自由度大。Ranking 比 absolute 好，但仍有
等價交換的部分（rank 6 跟 rank 7 的 block 如果 area 相同、連線相同，原圖
ordering 可能是隨機）。

如果這種「不可分辨 pair」佔 30%，rank_acc 上限就 70%，再怎麼訓也卡住。

---

## 🛠️ 訓練腳本指令參考

`iccad2026contest/training_example.py` 當前是 v3 ranking。所有 flag：

```bash
# Sanity check（~5 分鐘，不存檔、不污染 .pth）
python iccad2026contest/training_example.py --sanity

# 自訂 sanity 樣本數
python iccad2026contest/training_example.py --sanity --num-samples 200

# 正式訓練（預設 500 sample, ~30 分鐘）
python iccad2026contest/training_example.py

# 長時間訓練（建議搭配 --fresh 避免 cosine warm-restart 敲鬆 .pth）
python iccad2026contest/training_example.py --num-samples 2000 --fresh

# 關掉 aspect supervision，純看 ranking 效果
python iccad2026contest/training_example.py --num-samples 2000 --fresh --aspect-weight 0

# 自訂 aspect weight
python iccad2026contest/training_example.py --num-samples 2000 --fresh --aspect-weight 0.005
```

### Flag 預設值

| Flag | 預設 | sanity 預設 |
|------|------|------------|
| `--num-samples` | 500 | 120 |
| `--aspect-weight` | 0.001 | 0.0 |
| `--fresh` | False（嘗試 load v3.pth） | 同 |
| `--sanity` | False | True |

### 監控指標說明

| 指標 | 範圍 | 健康訊號 |
|------|------|---------|
| `rank` (loss) | [0, ∞), 隨機=0.693 | 持續下降 |
| `rank_acc` | [0, 1], 隨機=0.5 | 上爬到 0.7+ |
| `aspect` | [0, ∞), 觀察用 | 下降（即使 weight=0） |
| `lr` | cosine 0.001 → 0.00001 | — |

---

## 🚀 未來方向

### 路線 C-1: 加深架構（低風險）
- `n_gcn_layers` 4 → 6 或 8
- 加 multi-head attention 在 GCN 層之間
- 預期：rank_acc 上限提升 5-10%
- 風險：訓練時間增加 50-100%
- 觸發條件：先做 Q1 oracle 實驗確認 ranking 是 lever

### 路線 C-2: 改 ranking 目標（中風險）
- 目前 target = `fp_sol.x + fp_sol.y`（簡單但忽略結構）
- 改用更結構化的：
  - cluster_id × 1000 + local_rank → 同 cluster 的優先
  - 或 ListMLE 取代 pairwise BCE，捕捉全局排序
- 風險：可能跟 BL packer 的 expectation 不對齊
- 觸發條件：C-1 不夠用時再試

### 路線 C-3: 預測 adjacency matrix 或 tree_sol（高 ROI 但難）
- 訓練資料的 `tree_sol`（slicing tree，bsz × (n-1) × 3）**我們完全沒用過**
- 預測 slicing tree → 用標準 sequence-pair 或 B*-tree decoder 還原 layout
- 跟 BL packer 完全分開的路線，可能突破 v3 的 perm 限制
- 預期 ROI: 高（直接學「正確的 floorplan 結構」）
- 風險：需要重寫 decoder，1-2 週工程
- 觸發條件：v3 路線確定卡住

### 路線 D: 完全放棄 ML，純算法（最務實）
- 等組員的 `v8_puzzle_fingerprint_oracle.py`（目前缺檔）
- 1.0322 已驗證可達
- ML 上限可能差不多或更差
- ML 變成 algorithm 的補強，而非主力

---

## 📁 ML 相關檔案結構

```
FloorSet/
├── floorplan_gnn.pth                  ← v1 權重 (active, optimizer_claude.py 在用)
├── floorplan_gnn_checkpoint.pth       ← v1 訓練中 checkpoint
├── floorplan_gnn_v3.pth               ← v3 權重 (待訓練)
├── floorplan_gnn_v3_checkpoint.pth    ← v3 訓練中 checkpoint
├── predicted_floorplan_v3.png         ← v3 訓練尾端視覺化
└── iccad2026contest/
    └── training_example.py            ← 當前 v3 ranking 訓練腳本
```

**已刪除**：v2 的所有 artifact（commit `2235ad0`）。
**v1 沒刪**：因為 `optimizer_claude.py` 仍在用 v1 權重做 BL hint。

---

## 📜 歷史筆記（路線 A/B/C 演進）

最初的 gnn_training.md（v1 時代）規劃了三條路線：

- **路線 A**：深化神經網路（LR scheduler + weight annealing）→ 在 v1 階段
  實作了，幫助有限（loss 從 3.07 → 1.34，但 Total Score 進步 < 1%）
- **路線 B**：引入傳統 EDA legalizer 作後處理 → **完全沒做**，仍是可行方向
- **路線 C**（後來追加）：改 ML 預測目標 → 當前 v3 嘗試中

範式轉移後（2026-05-26），三條路線的優先順序變了：
1. **路線 D**（純算法）目前看起來 ROI 最高，但需要組員代碼
2. **路線 C-3**（預測 tree_sol）可能跟路線 D 互補
3. **路線 B**（legalizer）仍可作 ML 輸出的 post-processing
4. **路線 C-1/C-2** 是 v3 的微調，受 Q1 oracle 實驗結果限制
5. **路線 A** 已飽和，不建議再投入

---

## ⚠️ 給下一位維護者的提醒

1. **不要再用無監督 cost-formula loss 訓練 (v1 路線)** — 範式錯了，天花板 3.3
2. **不要再用 MSE on absolute position (v2 路線)** — ill-posed，必爆
3. **不要不做 oracle 實驗就直接 full training** — 沒方向感，浪費 GPU
4. **看到 rank_acc 卡 0.5 不一定是 bug** — 可能是 architecture 不夠，但也
   可能是 hyperparameter（aspect weight、LR scheduler、batch size）出問題
5. **不要動 `teammate_eva/` 內任何檔案** — 由使用者管理，是參考來源
