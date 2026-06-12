# ICCAD 2026 FloorSet — Session Context

## Claude 對話框規範
- 在聊天室中的語句必須**盡量精簡**。
- 在聊天室中請使用**繁體中文**回答。

## 🚨 範式轉移 (2026-05-26, 重要！)

**這個競賽不是 floorplan optimization，而是 reconstruction（拼圖還原）。**

### 證據（含 2026-05-27 修正）

組員的 1.0322 經查證 **是 oracle**（讀本地 validation label），hidden test 退回 fallback。
真正的 legit 上限約 **1.6**（無 label 的 portfolio 方法）。

- 組員 v8/v9 的 puzzle-fingerprint oracle 用 `FloorplanDatasetLiteTest` 讀**本地
  validation label**，fingerprint 比對輸入；命中回傳 ground truth，沒命中退回純算法
- 競賽 server hidden test 沒 label → 100% fallback，分數就是純算法 (`my_optimizer.py`) 的分數

組員從 `codex_experiment_log.md` 的演進軌跡（baseline=3.43，皆為 100% feasible）：

| 版本 | Total Score | Avg Cost | 性質 |
|------|------------|----------|------|
| baseline `my_optimizer.py` | 3.4292 | 3.1754 | 起點（與我們 3.33 同級）|
| v3 (boundary 強化) | 2.6919 | 2.5620 | 純算法 |
| v4 (擴及全 size) | 2.5816 | 2.1227 | 純算法 |
| **v5 (edge aspect 2.50/0.40)** | **1.7565** | **1.8299** | **單一最佳純算法** |
| v6 (7-profile portfolio) | 1.6204 | 1.6798 | runtime 10.6s |
| v7 (+MLP pose anchor) | 1.6174 | 1.6670 | runtime 11.2s |
| v8/v9 oracle (+repair) | **1.0322** | — | **要 label, hidden test 不適用** |
| v10/v11/v12 ML/clue 各種 | < 1% 進步 | — | 全失敗 |

→ **現實目標應該瞄準 ~1.6（legit portfolio），不是 1.03**
→ 我們 3.3258 vs 組員 legit 1.6 = **2× 差距**

### v5 的關鍵 insight：boundary aspect ratio (最高 leverage)

- LEFT/RIGHT-only blocks 用 aspect ratio **2.50**（高瘦）
- TOP/BOTTOM-only blocks 用 aspect ratio **0.40**（矮胖）
- 邏輯：邊界 block 拉高 edge capacity，減少 boundary violation
- 我們 `optimizer_claude.cpp` boundary block 預設方形 — **這是單一最高 ROI 改動**

### v10/v11/v12 ML 失敗教訓（避免重蹈）

- **v10 factor rank**：MLP 預測整數因子配對 aspect rank，val acc 44%，
  下游 < 1% → 一對多 + 機制不對齊
- **v11 clue chain**：用 b2b high-weight edge 拉連接 preplaced，80-99 退步
- **v12 contact plan**：把 b2b 接觸獎勵當第一順位，timeout 或大退步

→ 對我們 v3 ranking 的啟示：per-block local feature ML 在這題很弱，
   要動 ML 應該整合 *global* tiling/structure 訊號（contact graph、outline aspect 等）

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

### 🏆 最佳已驗證版本：Total Score = **1.3983** (Constructive portfolio M23: ORDER_MOVE relocation 軸 + 1 profile, 2026-06-12)

**反超組員所有 legit 版本（含 v6/v7 portfolio ~1.62，現 -13.7%）。** `constructive.cpp` +
`optimizer_constructive.py` 是組員 `my_optimizer.py` 建構式定框 floorplanner 的 C++
重寫（B 路線）+ 我們自建的 portfolio 選擇層。100/100 feasible，~9.5s/case（58
profile）。確定性（無 randomness/限時 → run-to-run 一致，可精確 A/B；官方 eval 1.3983）。

**M23（本 session，2026-06-12，接續 M22）= 兩個新 C++ 行為軸（ORDER_MOVE relocation
hill-climb + CLUSTER_ORD）+ 1 個 portfolio profile（57→58），驗證 -0.03%（1.3987→
1.3983，100/100 feasible，6/6 預估贏案全 realize — proxy = oracle 第七次驗證）+
om16 runtime 候補決策 + CLUSTER_ORD 死路驗證。**
- **🔑 ORDER_MOVE（`ICCAD_ORDER_MOVE=K`，constructive.cpp）**：在 ORDER_SWAP 之後的新
  jump move — 把 top-K total_wire item **拔出插到另一個 top-K 位置**（中間段順移一格），
  swap 做不到的結構移動（swap 固定其他人的位置）。同 OS 協議：pack-once 比較、
  layout_score 嚴格改善才收；accepted move 會位移後續位置 → 用 item 第一個 block id 追蹤。
- **ship om8_pin_wt_wire（OM8+PIN+WT+W2）+0.041%**：**case 89 歷來最深**（1.8155→
  **1.8061**，勝過 os32 候補的 1.8093）+ 57（1.4011→1.3689）+ 26/35/33/2。輕量
  （56 moves/frame，n=120 ~7s）→ avg runtime 8.79→9.49s/case，安全帶內。
- **⚠️ om16_bfs_wt_wire（OM16+BFS+WT 無 PIN）+0.148% 驗證有效但 runtime 回退**：
  **case 96（n=117）首殺**（1.3336→1.3160）+ **case 66 終於收割**（1.4378→1.3951，
  比 os24 候補 1.3989 深）+ 91/42/53/17/19。59-prof live = **1.3968、20/20 預估全
  realize**，但 n=120 cpu ~28s → avg 13.51s/case。**懲罰比公式：(13.51/8.8)^0.3 =
  +13.8%（與 median 值無關，只要 median < ~19s 必然成立）換 +0.14% → 回退**，與
  os24/os32 同列候補（**候補合計 +0.23% 現成**，等官方 runtime 規則確認寬鬆）。
- **K 放大與排序的交互（同 OS 軸 pattern）**：K=8→16 在 BFS+WT 上 +0.026%→+0.148%，
  在 PIN+WT 上 +0.041%→+0.040%（無效）；**OM12 完全丟失 96/66（僅 +0.012%）— hill-climb
  路徑對 K 非線性，無便宜折衷**。OM8+OS16 疊加 +0.070%（case 96→1.3242/45/53/26）被
  兩個主候選蓋過且 runtime 同級重 → 不加。
- **❌ 死路 6：CLUSTER_ORD（cluster 複合 item 在 bscore 類內最前/最後，
  `ICCAD_CLUSTER_ORD=1/2`）**：兩個排序上全 0.000%（唯一贏案 case 17 n=38 權重 ~0）。
  code 保留（env-gated）勿重掃。OM8+BFS+tight 也 0.000%（OS8+tight 曾 +0.221%，
  但 OM 在 tight 上無效 — move 與 swap 的有效排序面不同）。
- ⚠️ 下一步：OM 軸已收割（om8 ship、om16 候補、K×排序組合掃完）→ 剩餘新 C++ 行為軸：
  refinement pair-relocation（placement 層，非 order 層）、compaction 方向偏好 /
  pack over-spread 軸先、pack 向 connectivity 重心。case 89 殘留 1.8061、85 1.6255、
  96 1.3336（om16 可到 1.3160）、66 1.4378（om16 可到 1.3951）。

**M22（前一 session，2026-06-11，接續 M21）= OS K 放大掃描收尾 + 2 個 portfolio profile
（55→57），驗證 -0.08%（1.3998→1.3987，100/100 feasible，8/8 預估贏案全 realize）+
K 軸飽和證明 + runtime 風險決策。**
- **🔑 K 軸在最強排序上於 K=16 飽和**：OS24（+0.041%）與 OS32（+0.062%）的贏案裡
  **case 98/79/82 全部絕跡** — K=16 已拿走高權重案的 jump-move 紅利，更大 K 只在中型案
  各撿不同的渣（hill-climb 路徑不同 → OS24 拿 66、OS32 拿 71，互不重疊）。
- **ship os16_bfs_wt_wire（OS16+BFS+WT 無 PIN）+0.056%**：case 86 再磨深（1.3347→
  1.3255）+ 82（1.4679→1.4565）+ 50（1.2920→1.2419）+ 29。
- **ship os16_bfs_tight_wire（OS16+BFS+tight）+0.057%**：**新 case 62 首殺**（1.6214→
  1.5248）+ 55（1.8198→1.7828）+ 51 + 66 半收（1.4450→1.4378）。
- **⚠️ runtime 風險決策（重要先例）**：os32_pin（case 71 1.3187 + 最深 89 1.8093）與
  os24_pin（**唯一拿下長期未收割 case 66 → 1.3989** + 27）四個全加 live 驗證 =
  **1.3979（掃描預估完全吻合、100/100 feasible）但 avg runtime 8.4→21.9s/case（2.6×）**。
  官方 RuntimeFactor = max(0.7, R^0.3)，R 分母是 cross-submission median（未知；組員
  portfolio ~11s 是唯一參考）→ 21.9s 若 median 8-12s 會吃 +20-35% cost 懲罰，遠超
  0.04% 分數差 → **回退兩個重 profile**（n=120 cpu：OS24 32s、OS32 58s）。候補記錄在
  optimizer_constructive.py 註解，**若官方 runtime 規則確認寬鬆可加回（+0.08% 現成）**。
- wrapper subprocess timeout 55→**120s**（防呆上限，OS16 變體在 57-prof 並行下大案
  wall 可能 >55s；無壞處）。
- ⚠️ 下一步：**OS K 軸枯竭**（K=16 飽和 + K>16 runtime 不划算）→ 回到新 C++ 行為軸：
  cluster ordering 變體、refinement pair-relocation、compaction 方向偏好、pack
  over-spread 軸先。case 89 殘留 1.8155（os32 可到 1.8093）、85 1.6255、66 半收 1.4378
  （os24 可到 1.3989）。

**M21（前一 session 稍早，2026-06-11，接續 M20）= ORDER_SWAP 組合掃描 + 3 個 portfolio profile
（52→55），驗證 -0.58%（1.4080→1.3998，100/100 feasible，12/12 預估贏案全 realize，
掃描預估 1.3999 vs 實際 1.3998 — proxy = oracle 第五次驗證）。突破 < 1.40 目標。
無 C++ 改動，純 portfolio 層（M20 的 ORDER_SWAP 機制 × 新 K/排序/frame 組合）。**
- **🔑 os16_pin_wt_wire（OS16+PIN+WT+W2）+0.460% = M18 BFS 以來最大單 profile**：更大
  swap 池（top-16 = 120 對 vs top-8 = 28 對）疊在最強排序上 — **最高權重 case 98 再殺**
  （1.4118→**1.3841**）+ **case 79 再破**（1.5135→**1.4219**，M17 起四連降）+ 82
  （1.4941→1.4679）+ **硬 case 89 首次鬆動**（1.8273→1.8155）+ 91/47。**K=8→16 增益
  未見頂** — 對 jump-move 池加深比換排序起點值錢。
- **os_bfs_wt_wire（OS8+BFS+WT+W2 無 PIN）+0.262%**：case 86（1.3775→1.3347）+ 95
  （1.2767→1.2598）+ 97（1.2779→1.2713）。PIN 與否改變 BFS seed → swap 起點不同 →
  與 PIN 版贏案幾乎不重疊。
- **os_bfs_tight_wire（OS8+BFS+tight+W2）+0.221%**：**硬 case 85 首次下殺**（1.6606→
  1.6255）+ 42（1.5154→1.3990）+ 40（1.5804→1.4609）。三個新 profile 贏案近零重疊。
- **不加**：OS12+PIN（+0.157%，case 89 1.8106 最深但其餘全被 OS16 蓋過，殘留 ~0.017%）、
  OS16+BFS+tall（+0.056%，89/97 全被蓋過殘留 ~0）、OS8+WT+tall（+0.016%）。
- runtime 5.10→**8.36s/case**（55 profile；K=16 的 120 對 swap pack 是主因。本地 eval
  RuntimeFactor=1.0 中性；再加重 profile 留意官方 cross-submission median）。
- ⚠️ 下一步：**OS K 繼續放大**（OS24/OS32 on PIN+WT — K=8→16 給 +0.46% 邊際未見頂，
  swap 段 runtime ~2.3×/4.4×）；case 89 殘留 1.8155（os12_pin 可到 1.8106 差 0.017%
  單獨不過門檻，找「拿 89+其他」組合）；case 66（1.4450）仍未收割。

**M20（前一 session，2026-06-10，接續 M19）= ORDER_SWAP pack-order pair-swap hill-climb +
1 個 portfolio profile（51→52），驗證 -0.18%（1.4105→1.4080，100/100 feasible，12/12
預估贏案全 realize）+ 兩個軸死路驗證。**
- **🔑 ORDER_SWAP（`ICCAD_ORDER_SWAP=K`，constructive.cpp）**：每 frame 在 refinement 前，
  對 **top-K total_wire item 的 28 對（K=8）做 pack-order 互換 hill-climb** — 每 swap 重
  pack 一次、layout_score 嚴格改善才收。這是 greedy 排序 + force-directed refinement 做不到
  的 **jump move**（greedy 一旦放錯位，refinement 只能微調不能重排）。pack-once vs
  pack-once 公平比較、下檔保護、refinement 從更好的 order 起步。items 在 frame loop 間
  用 items_base 隔離（每 frame 從乾淨 order 開始）。
- **加 1 個 profile：`os_pin_wt_wire`（OS8+PIN+WT+W2）+0.252%** — 疊在最強排序上：
  **case 94 再殺**（1.3411→**1.3128**）+ **case 98 再殺**（1.4221→**1.4118**）+ 試金石
  **case 79**（1.5249→1.5135）+ 50/28/9/4/5/2/12/18。
- **OS 需要好的起始 order 才在高權重案有效**：OS8 單獨 +0.030%（12 個贏案全低權重，
  42/40 等「無人贏過」的中型案）、OS8+BFS+tall+W2 +0.025%（case 89 微降 1.8273→1.8241）
  → 都不過門檻不加，但顯示 swap 對中型案有廣泛弱效果。
- **❌ 死路 4：BFS_NORM（attach/√area 正規化）**：4 個組合全 ≤0.026%（最佳僅 case 66）。
  code 保留（env-gated）勿重掃。
- **❌ 死路 5：PIN 補掃（PIN+tight/PIN+narrow）**：≤0.030%（也是 case 66）。**case 66
  （1.4450，多個變體可到 ~1.389）是反覆出現但單案 ~0.03% 不過門檻的未收割案**。
- runtime 4.48→5.10s/case（52 profile，OS profile pack 次數 ~3×；RuntimeFactor=1.0 中性）。
- ⚠️ 下一步：OS 軸剛開 — OS+bfs_wt_wire（無 PIN）/ OS+tall 組合 / K=12/16 未掃；
  case 89（1.8241）仍最高，case 53（1.4266，proxy 反覆 miss 0.0009）可查。

**M19（同日稍早，2026-06-10，接續 M18）= BFS_PIN pin-anchored seed 變體 + 2 個 portfolio
profile（49→51），驗證 -0.23%（1.4138→1.4105，100/100 feasible，13/14 預估贏案 realize，
唯一 miss case 53 差 0.0009 / 權重 0.000%）。**
- **🔑 BFS_PIN（`ICCAD_BFS_PIN=1`，constructive.cpp）**：WIRE_BFS 的初始 attachment 原本
  只算 preplaced b2b 連接 — 但 **pins 的位置同樣固定**（greedy wire 項看得到它們）→ 把
  p2b pin 權重也加進 BFS seed attachment。pin-連接強的 item 更早放 → 在 p2b-heavy case
  上 BFS 順序更貼近真正的「已知位置」結構。一行擴充，疊在 WIRE_BFS 上。
- **加 2 個 profile（掃 11 個候選：8 個 BFS knob 組合 + 3 個 PIN 變體）**：
  - `bfs_pin_wt_wire`（PIN+WT+W2）**+0.269%**：**case 95 再下殺**（1.2995→**1.2767**，
    M18 後又 -1.8%）+ **case 94**（1.3656→1.3411）+ 64
  - `bfs_tight_wire`（BFS+tight+W2）**+0.061%**：case 91（1.3848→1.3712，PIN 蓋不到）
    + 43/8/1/3/12 等小案
  - 不加：BFS+anc+W2（+0.063% 但唯一主力 case 94 被 PIN+WT+W2 蓋過）、PIN+W2（+0.126%，
    殘留只剩 case 66 +0.029%）、PIN+tall+W2（+0.098%，殘留 ~0）、其餘 BFS 組合 ≤0.025%
    （**BFS knob 組合軸掃尾完成**：asp5/asp7/WT+tall/narrow+anc/WT+narrow/WT+asp5 全死）
- runtime 4.51→4.48s/case（51 profile，中性）。
- ⚠️ 下一步：PIN 軸只掃了 3 個組合（PIN+tight / PIN+narrow / PIN+anc 未掃，但 BFS knob
  軸已枯竭 → 期望低）；case 66（1.4450，PIN+W2 能到 1.3891）是已知未收割最大單案；
  更深結構：refinement pair-relocation、BFS attachment 正規化（wire/area 比）。

**M18（同日稍早，2026-06-10，接續 M17）= WIRE_BFS pack-order 新軸 + 3 個 portfolio
profile（46→49），驗證 -0.45%（1.4202→1.4138，100/100 feasible，16/16 預估贏案全
realize、殘留 0.000% — proxy = oracle 持續成立）。M14 以來最大單 session 增益。**
- **🔑 WIRE_BFS（`ICCAD_WIRE_BFS=1`，constructive.cpp 新排序層）**：在 base/WT 排序之上的
  **類內 BFS-connectivity 重排** — bscore 類邊界不動（WIRE_ORDER 跨類教訓），類內每步
  greedy 選「與已排 items（任意類）+ preplaced blocks 連接權重最大」者先放；attachment
  平手保基序（無 wire 訊號時退回 base/WT 行為）。效果：greedy wire 項幾乎總看得到 item
  重邊的「已放端」，早期 item 不再盲放。獨立 if 疊加層 → 可與 WT（決定 tie 序）、frame
  shape 任意組合。
- **加 3 個 profile**（離線掃 8 個 WT 組合 + 6 個 BFS 變體，>0.05% 才加）：
  - `bfs_wt_wire`（BFS+WT+W2）**+0.316%**：最高權重 **case 98** 再下殺（1.4413→1.4221）
    + 95（n=116，1.3215→1.2995）+ 91/50/32/27/18/1/5
  - `bfs_tall_wire`（BFS+tall+W2）**+0.251%**，與上面**贏案零重疊**：**case 79 再破**
    （1.5974→1.5249，M17 破解後又 -4.5%）+ 86/74/97/89
  - `wtb_tall_anc`（WT+tall+anc，M17 未掃組合）+0.071%：**case 87 獨有**（1.4512→1.4272）
- **WT knob 軸枯竭確認**：其餘 WT 組合（anc_lo/tight/asp7/W2+anc/tight+W2/asp7+W2）全
  ≤0.03%；BFS 單獨 +0.150%、BFS+narrow+W2 +0.131% 的贏案幾乎全被 ship 的兩個 BFS profile
  蓋過 → 不加。
- runtime 3.86→4.51s/case（49 profile，RuntimeFactor=1.0 中性）。
- ⚠️ 下一步：BFS 組合空間只掃了 6 個；未掃 BFS+tight、BFS+asp 系列、BFS+WT+tall、
  BFS+narrow+anc 等。case 79 殘留 1.525、case 89 殘留 1.827 仍是最大單案。

**M17（同日稍早，2026-06-10，接續 M16）= WIRE_TIEBREAK pack-order 新軸 + 2 個 portfolio
profile，驗證 -0.20%（1.4231→1.4202，100/100 feasible）+ per-frame csc 死路驗證。**
- **🔑 WIRE_TIEBREAK（`ICCAD_WIRE_TIEBREAK=1`，constructive.cpp 新 item 排序變體）**：
  bscore（boundary 優先）仍是第一鍵不動，但同 bscore 類內**把 total_wire 最大的 item 先放**
  （取代 size 鍵）→ greedy wire 項早期就看得到重連接鄰居。⚠️ WIRE_ORDER（wire 當第一鍵）
  曾失敗（vBd 390）；tie-break 變體保住 boundary 優先級 → vBd 不爆。
- **加 2 個 profile（44→46）**：`wtb_wire`（WT+W2）、`wtb_tall_wire`（WT+W2+tall frame）。
  離線掃描（profile_vs_portfolio.py）：wtb_tall_wire oracle-min **+0.216%** — **破解 case 79**
  （先前判定「無 cheap trick」的最大 hgap 單案，1.706→**1.597**；它是 dense uniform graph，
  正需要 wire-driven 聚集 — 診斷正確引導了解法）+ case 89/53/16；wtb_wire +0.084% 獨佔
  **最高權重 case 98**（1.4502→1.4413）+ 63。其他組合（narrow/LR5/W3 版）贏案全被這兩個
  蓋過 → 不加。live eval **1.4202**（-0.20%，兩 profile 合併 realize）。
- **❌ 死路 3：per-frame compaction + csc 重估 frame 選擇（原最高 ROI 候選）已實測無效**：
  把 compaction 從「只對 layout_score winner 做」改成「每個 frame finalist compact+push 後
  用 csc 選最終」。(a) default-on：單 base 1.5197→**1.5293 退步**（vBd 270→290、vCl 62→50 —
  csc 固定 hw 權重在跨 outline 比較時失準，拿 cluster fragment 換 boundary violation，
  M10 警告的變體重現）；(b) 當 portfolio profile：oracle-min 僅 **+0.008%**（1 case）。
  **跨 frame/layout 選擇本來就是 wrapper shapely proxy 的工作**（它已跨 46 profile 做同樣的
  事且更準）。實驗 code 已移除（比照 preplaced-frame 先例），`csc_of` helper 重構保留。
- **❌ 同場死路：NO_COMPACT profile**（0.000%）、WT 單獨（+0.017%）、WT+W3（+0.004%）。
- ⚠️ 下一步：WIRE_TIEBREAK 軸還可掃更多組合（anc_lo / tight / asp 系列 × WT）；case 79 cost
  仍 1.597（hgap 殘留）可再攻；或繼續結構性方向（見「下一步」）。

**M16（同日稍早，2026-06-10，接續 M15 攻 hgap）= HPWL push 加 same-size swap，
default-on，驗證 -0.04%（1.4236→1.4231，100/100 feasible）+ 兩個重要死路驗證。**
- **同尺寸 swap（唯一 ship 的正向改動）**：`hpwl_push()` 內新增 swap pass — 兩個
  **bit-identical (w,h) 且相同 boundary code** 的 non-preplaced non-cluster block 交換位置，
  嚴格降 HPWL 才接受。**幾何 multiset 完全不變** → bbox/area、每邊觸碰數（bv）、cluster
  幾何（gf）、dims（mib）全部 identical by construction → downside-free，default-on。
  同 area + 同 boundary code 的 soft block 經 `default_soft_dim` 得 bit-identical dims →
  exact-equality 分組找得到真 partner。單 base A/B：1.5202→**1.5197**（vBd/vCl/vMb 維持
  270/62/0 不變）；Python 原型（dbg_hpwl_push.py ENABLE_SWAP）-0.03%、0/100 退步；C++ 全
  profile + proxy 重選 → **1.4231**。env: `ICCAD_NO_SWAP=1` 退回 M15。
- **❌ 死路 1：violating boundary single 修復（原「下一步 6」，最高 ROI 提案）已實測無效**：
  原型（dbg_hpwl_push.py `ENABLE_VIOLATING`）做「沿 constrained 軸推回 frozen-bbox 邊修
  violation（無條件接受），推不到邊則未釘軸 median 滑（嚴格降 hpwl）」。**真值結果 0 個 bv
  修好、delta -0.00%**。`dbg_vio_stats.py`（新工具，對 portfolio JSON 分類）揭示根因：100 案
  202 個 violating boundary block 中 **123 個是 cluster 成員**（不可動，M10 精度牆）、45 個
  preplaced（固定）、**只有 34 個 single 且全部 BLOCKED**（推到邊必撞別的 block）→ 可修數
  = 0。residual vBd 結構上不是後處理能修的。原型旗標保留（預設 False），勿重試。
- **❌ 死路 2：profile knob 軸枯竭**：`profile_vs_portfolio.py` 掃 5 個候選 —
  WIRE_MULT=4.0（最佳，oracle-min 僅 +0.063%）、WIRE_MULT=6.0（+0.018%）、LR3.5+W4（0.000%）、
  ANCHOR_W=0.30（+0.011%）、ultra-narrow frames 0.50-0.22（+0.002%）。**env 旋鈕組合變體
  路線到頭**，再加 profile 只花 runtime 不動分。
- **case 79（hgap 1.015 最大單案）診斷結論：無 cheap trick**。它是 dense uniform graph
  （700 edge、mean degree 14、weight 全 ~0.003 均勻）→ HPWL ≈ 所有相連 pair 平均距離；
  baseline 平均 pair 距離 ~49 vs 我們 ~81（hbase=105 異常低，hmin/hbase=1.51）。要壓它需要
  全局 wire-driven 聚集（placer 結構），portfolio selection 已無空間（proxy pick 1.7072 vs
  oracle 1.7063）。
- ⚠️ push/swap 微調路線已收割完（M14 -0.67% → M15 -0.12% → M16 -0.04%，遞減）→ **下一步
  轉攻結構性改動**：per-frame compaction + csc 重估 frame、wire-driven packing 改進（見「下一步」）。

**M15（前一 session，2026-06-08，接續 M14 攻 hgap）= HPWL push 擴大可移範圍 → boundary-axis
slide，default-on，驗證 -0.12%（1.4253→1.4236，100/100 feasible）。**
- **目標**：M14 free-single push 後 weighted hgap 仍 0.404（最大殘留）。CLAUDE.md「下一步 5」
  要擴大 push 可移範圍到 (a) cluster-rigid、(b) boundary-axis slide，兩者皆宣稱 downside-free。
- **做法 `hpwl_push()`（constructive.cpp）**：把可滑 block 從只有 FREE SINGLE 擴到 **BOUNDARY
  SINGLE**（boundary≠0 ∧ cluster==0 ∧ 非 preplaced ∧ 非角塊）。boundary block 剛好一軸被邊
  釘住（LEFT/RIGHT 釘 x、TOP/BOTTOM 釘 y）→ **只滑 free 軸**（LR 滑 y、TB 滑 x），constrained
  軸座標不變 → edge-contact（故 bv）保住。x/y 滑動邏輯重構成 `slide_x`/`slide_y` lambda 共用。
- **🔑 為何仍 downside-free（可證）**：滑動留在 bbox 內 → bbox 非增。bbox 縮只會 un-satisfy
  「縮掉那邊」的 boundary block，但**已滿足的 boundary block 正好貼在那邊 → co-define 該極值 →
  釘住該邊，不可能被 un-satisfy**。故 area/bv/gf/mib 全不變，只降 hpwl（且每步嚴格降才接受）。
  單 base A/B 證實：**1.5219→1.5202，vBd/vCl/vMb 維持 270/62/0 完全不變**。Python 原型
  （`dbg_hpwl_push.py`，對 evaluate_solution 真值）+boundary-axis **-0.11%、0/100 退步、
  0/100 bv-or-gf 增加**；C++ 全 profile push + proxy 重選 → **1.4236（比原型 1.4237 略低**，
  proxy 在 pushed 候選中重選更佳）。env: `ICCAD_NO_BND_PUSH=1` 退回 M14（只滑 free single）。
- **❌ cluster-rigid（選項 a）已試並棄用**（`dbg_hpwl_push.py` 的 `ENABLE_CLUSTER`）：
  compound-item packer 把 cluster 塞得**沒有 slack**（100 案中只有 1 個 free cluster 能動），
  且 FP 剛體平移 `pos[m]+=d` 會在 ULP 級**破壞 cluster 內部精確 abutment** → shapely（精確）
  判為虛假 grouping fragment（M10 精度陷阱重現）。淨效果 **+0.004% 且 1 退步** → 無價值，不實作。
  教訓：任何移動 cluster 成員的後處理都會撞 M10 精度牆，除非重建精確 abutment。
- ⚠️ ~~下一步：boundary 沿 constrained 軸修 violating block、case 79 個案~~ → **M16 已驗證
  兩者皆死路**（violating 全 blocked；case 79 是 dense uniform graph 無 cheap trick，見 M16 段）。

**M14（前一 session，2026-06-07，攻 hgap = 最大 cost lever）= post-placement HPWL push，
default-on，驗證 -0.67%（1.4349→1.4253，100/100 feasible）。**
- **診斷**：portfolio per-case gap 分解（讀 results.json 的 hpwl_gap/area_gap/
  violations_relative，加權 e^(n/12)）→ **weighted hgap=0.412 ≫ agap=0.228 ≫ vrel=0.040**。
  cost=`(1+0.5(hgap+agap))·exp(2vrel)` → **HPWL gap 是壓倒性主 lever**（2× area、10× viol）。
  最高權重 case 93-99（~40% 權重）hgap 全在 0.31-0.44。
- **根因**：placer 沒有任何 post-placement HPWL 優化。compaction 只往 frame 四面 pack（攻
  area），反而把相連 block 拉往對面 → **抬 HPWL**。
- **做法 `hpwl_push()`（constructive.cpp）**：compaction 後，對每個 **FREE SINGLE** block
  （boundary==0 ∧ cluster==0 ∧ 非 preplaced）沿 connectivity 加權 **L1-median** 滑進
  current bbox 內的 void（coordinate descent、Gauss-Seidel、PUSH_PASSES=8）。void 區間
  `[lo,hi]` = 另一軸 overlap 的最近 block 邊界 → block 留在自己 slot，**可證不產生 overlap**。
- **🔑 為何 default-on（不需 portfolio 下檔保護）= 構造上 downside-free**：free single 不貢獻
  boundary/grouping violation 且滑動留在 bbox 內 → **area / bv / gf / mib 全不變，只降 hpwl**
  （且每步只在嚴格降 hpwl 時才接受）。單 base A/B 證實：1.5306→1.5219，**vBd/vCl/vMb
  維持 270/62/0 完全不變**。Python 原型（`dbg_hpwl_push.py`，對 evaluate_solution 真值）
  -0.63%、**0/100 退步**；C++ 全 profile push + proxy 重選 → **1.4253（比原型 1.4259 更低**，
  因 proxy 在 pushed 候選中重選更佳）。env: `ICCAD_NO_PUSH=1` 關閉、`ICCAD_PUSH_PASSES=N`。
- **proxy 在 pushed set 上仍完美**：`rh_sweep.py` 確認 oracle-min（完美選擇）= **1.4253**，
  官方 eval live 也 = 1.4253 → **proxy = oracle ceiling**（_RH=1.4 仍在平坦盆地 1.1-1.6，無需重調）。
- runtime 影響可忽略（8 pass × N²/profile，N=120 約 5M ops/case），eval RuntimeFactor=1.0 中性。
- ⚠️ push 後 hgap 0.412→0.404（仍是最大殘留）→ **下一步繼續攻 hgap**：把可移範圍擴到
  cluster-rigid（整 cluster 一起滑，保持 grouping）、或 boundary block 沿邊軸滑（不破 edge-contact）。

**M10（前一 session，2026-06-06，攻 area_gap dead space）= 兩件事，皆驗證有效：**

1. **🔑 輸出精度修正 `%.10f`→`%.17g`（最大單一 lever，-6.3% 單 base）。** placer 的
   cluster compound-item（及 compaction packs）造**精確 abutment**（A 右緣 == B 左緣，
   gap=0）。`%.10f` 捨入可移座標 ~1e-10，開出 sub-nm 縫；C++ union-find（1e-3 tol）看不到，
   但 **shapely（精確）當成 cluster fragment** → 虛假 grouping violation → cost 灌水。改
   `%.17g`（IEEE double 完美 round-trip）→ 縫消失。**這是既有 pipeline 一直在漏的分**
   （~144 個虛假 fragment / 100 案）。單 base **1.658→1.5532**。⚠️ 任何 C++ 吐給 shapely
   評分的幾何都要 %.17g；查不明 grouping violation 先疑精度再疑 packing。
2. **boundary 接觸保持的 compaction（再 -1.3% 單 base → 1.5335）。** `compact_layout()`：
   四個 order-preserving 方向 pack（左/右/下/上，**可證無重疊**、preplaced 釘死、保序），
   re-snap 單 boundary block，取最佳候選。pack 向某面也把散開的 cluster 成員拉成 abutment
   → **連帶降 grouping fragment**（vBd 287→272、vCl 74→61）。對選最佳的單一 frame 套用一次
   （非 per-frame：餵 layout_score 每個 frame 的 compacted 變體會 overfit proxy）。
   ⚠️ **選候選用 true-cost proxy `csc=(area+hw·hpwl)·exp(2·(bv+gf)/nsoft)`，不可用
   layout_score**：後者 boundary 權 150000、grouping 僅 6500（23×），會愛上「拿 boundary
   miss 換 cluster fragment」的 pack，但真 cost 把 (bv+gf+vmb)/nsoft 丟進 exp() → 該交易中
   性（且常抬 hpwl 反虧）。csc 等權 bv/gf 並除以 nsoft，case 99 因此正確保留原圖。
   env: `ICCAD_NO_COMPACT=1` 關閉。工具：`dbg_compact.py`（Python 原型，對 evaluate_solution
   真值；**選擇 proxy 與輸出精度必須對齊 C++**否則高估）、`dbg_compact_cmp.py`（單案 Py vs
   C++ 對照，當初靠它抓出精度 bug）。

**綜合**：單 base 1.658→**1.5335（-7.5%）**；portfolio **1.5362→1.4528（-5.4%）**。

**M13（2026-06-07，narrow frame profile + proxy _RH 修正）= 兩件事，第二件是最大 lever：**

1. **narrow-frame profile（40→44 profile）。** `dbg_area` 顯示 dead space 是**系統性水平**
   的：`w/wb`（我們寬/baseline 寬）普遍 1.3-1.5、`h/hb`≈1.0 → **我們 pack 太寬**。加 4 個比
   frame_tall（aspect 0.67-0.33）**更窄**的 outline profile（aspect **0.55-0.28**：narrow /
   narrow_wire_anc / narrow_wire / narrow_anc）。逐案真值比較確認：narrow **贏下最高權重的
   case 98（n=119）與 87**（frame_tall 搆不到）。但單獨加 narrow 只到 **1.4369**（≈ 沒動）—
   因為 proxy 選不出來（見下）。
2. **🔑 proxy `_RH` 1.0→1.4（最大 lever，realize 整個 oracle ceiling）。** 診斷 case 98：
   proxy 選 LR3.5（真 cost 1.4892），但真正最佳是 narrow（1.4697）。根因：true cost =
   `0.5·(hpwl/hbase + area/abase)·exp(2·vrel)`，而 proxy 用 **hmin（各 profile 最小 hpwl）
   當 hbase 的替身**。我們永遠贏不過 baseline hpwl → `hmin > hbase`，比值 `hmin/hbase ≈
   1.3-1.4`（case 98 418/308=1.36、96 1.38、94 1.30，**跨案集中**）→ raw proxy **低估 hpwl
   項**，選不出 hpwl 最低的 narrow。對策：proxy 的 hpwl 權重 `_RH` 從 1.0 提到 **1.4** 補償。
   `rh_sweep.py` 離線掃描（4200 profile×case 真值快取，offline 掃 _RH）：**_RH 1.3-1.6 是平坦
   盆地，全部命中 oracle ceiling 1.4349**（_RH=1.0 才 1.4369）→ 泛化安全（非尖峰）。選 1.4
   （盆地中心 + 等於 principled hmin/hbase 均值）。

**綜合**：portfolio **1.4371→1.4349（-0.15%）**，100/100 feasible，4.59s/case。**proxy 現在 =
oracle ceiling**（44 profile 完美選擇）→ selection 不再是瓶頸，往後加 profile 全額 realize。
⚠️ 試過失敗：preplaced-aligned frame（攻 case 89）— greedy packer pack 不下 tighter width，
case 89 結構性無解，且 incidental 贏的案全被現有 profile 蓋過（零貢獻）。見下方 M12 段尾。
工具：`rh_sweep.py`（建真值快取 + 掃 _RH/proxy 參數，最快 proxy 調參器）、`proxy_dbg.py
<ids>`（單案逐 profile proxy vs 真 cost，找 mis-selection）、`profile_vs_portfolio.py`。

**M12（2026-06-07，組合 profile 擴充）**：`optimizer_constructive.py` 從 14 擴充到 **40
profile**（+26 個組合 knob 變體）。新 profile 涵蓋以前缺失的組合：LR_ASPECT × WIRE_MULT
× ANCHOR_W 三路組合、frame_tall × aspect 組合、frame_tight × aspect 組合等。Portfolio
下檔保護確保無用 profile 不傷分（proxy 精確 → 只有真正更好的 profile 才被選）。最大貢獻
組合：`tall_anclo`（frame_tall+anc_lo）、`asp5_all`（LR5+anc_lo+WIRE×2）。
14→16→18→21→24→27→30→40 profile 遞增，每批 eval 確認正方向：
1.4502→1.4473→1.4455→1.4447→1.4444→**1.4387**→1.4378→**1.4371（-0.87%）**。
runtime 14→40 profile：1.46s→4.0s/case（並行），contest RuntimeFactor=1.0 → 分數無影響。
⚠️ 試過失敗的 profile 類型：`wire_all`（bp>0 也算 wire → regression + 2× 慢）、
`wire_order`（wire 排序 → vBd 390，退步 1.8605）。
**portfolio 1.4502→1.4371（-0.87%），突破 < 1.43 目標。**

**M12 後（2026-06-07）試過失敗：preplaced-aligned frame（option A，攻 case 89）。**
診斷 case 89（cost 1.848，最高）：3 個 FREE preplaced RIGHT block 在 x=142，但 movable
block（16/34/67/78/97）pack 到 145.03 → bbox 右緣 145 > 142 → preplaced 達不到右邊界 →
3 個虛假 vBd。對策：加「width 釘在 pre_w(=142) 的 frame 候選」，逼 movable 不超出 preplaced
外緣。**失敗，已 revert**：(1) greedy packer **packtight 不下** width 142（即使 frame 加高
到 total×2.0，pack_in_frame 仍回 false → case 89 完全不動，bv 維持 7）；(2) 把 pinned frame
強制 always-try 後，layout_score 因 150000·bv 權重愛上 pinned frame（case 85/99/80 退步，
bv 降但 agap 爆），但 case 89 仍不動；(3) 唯一「贏」的 case（79/88/84/67/74/46）**全部已被
現有 aspect/frame profile 在 portfolio 蓋過**（逐案比 portfolio cost：preplaced-frame 無一
勝出）→ portfolio 淨增益 **= 0**。**結論：case 89 結構性無解（packer 牆），preplaced-frame
對 portfolio 零貢獻。** env 旋鈕 `ICCAD_PREPLACED_FRAME` 與相關 code 已全部移除。
工具：`profile_vs_portfolio.py`（跑任意 env profile，逐案比 portfolio JSON cost，算 oracle-min
增益 — 通用「新 profile 值不值得加」測試器，用法 `profile_vs_portfolio.py KEY=VAL ...`）。

**M11（2026-06-07，迭代 compaction）**：在 M10 的 12-candidate 初始輪之後，繼續從當前最佳
迭代跑單軸 pack（pack_x→pack_y→pack_x… 最多 `COMPACT_ITERS=8` 輪）直到 csc 不改善為止。
Y-pack 位移行後 X-pack 可回收新 slack；反之亦然。csc 下檔保護，deterministic。
收斂通常 1 輪（初始 12-candidate 已含 2-step combo 挑過大多數增益）。
env: `ICCAD_COMPACT_ITERS=N`（N=0 等同關閉迭代輪次）。
**單 base 1.5335→1.5306（-0.19%）；portfolio 1.4528→1.4502（-0.18%）**。
⚠️ cluster-rigid compaction（整 cluster 當剛體一起滑）已試並**失敗**（1.5306→1.5464）：
rigid shift 對 csc 似乎改善但 true evaluator 退步，已 revert。root cause 未確認。

**M9（前一 session）= two-pass wire refinement。** 主因：

**M9（本 session，2026-06-06，強化單一 placer）= two-pass wire refinement。** 主因：
最大 cost 缺口是 **HPWL gap（hgap ~0.4-0.6）**，而 greedy 的 wire 項只看「已放」鄰居 →
最早放的 block 幾乎是盲放（anchor 只來自 preplaced）。對策：每個 frame 先正常 pack 一次，
再用上一輪位置當「guide」重 pack `REFINE_ITERS=12` 次，每次 wire 項額外拉向「尚未放」鄰居
的 guide 位置（force-directed coordinate descent），逐輪推進 guide 收斂，per-frame 取
layout_score 最佳（下檔保護、確定性）。單 base 1.7045→1.658（-2.7%，vBd 307→287），
portfolio **1.5659→1.5375（-1.8%）**；proxy selector 未退化（live 緊跟 ceiling）。
env: `ICCAD_NO_REFINE=1` 關閉、`ICCAD_REFINE_ITERS=N`（單 base 12-24 已平緩 ~1.655）。
本地 runtime 對分數中性（eval 強制 RuntimeFactor=1.0），故 2× runtime 安全。
⚠️ 試過無效：`layout_score` 的 hpwl 權重（HW_MULT 3/8 完全不動選擇 → area 項主導）。

**M9 後 area 分析（`dbg_area.py`，重要結論）**：refinement 攻完 HPWL 後，最大殘留是
**area_gap ~0.266**。分解發現它**幾乎全是可移除 dead space**：原圖 density（bbox/ΣblockA）
= **1.035**（緊），我們 = **1.311**（~27% void）。且 dOurs 大量卡在 **1.323 = 1.15²** →
packer 一直退回 s=1.15 frame，因為 s=1.05 tight frame **pack 不下**（greedy 留 void）。
- 試 finer frame scales（單 base agap 0.266→**0.196**）→ 但 vBd 287→**316**（tighter pack
  把 boundary block 擠離邊）→ 單 base score 反退 1.6914。**area↔violation tradeoff**。
- 故加成 **frame_fine** profile（scales 1.04-1.16 + 大 backstop，下檔保護）→ portfolio
  1.5375→**1.5362（僅 -0.08%）**。**結論：frame 路線的 area 已枯竭** — tighter outline 必
  加 violation，被 `layout_score` 的 150000·bv 擋掉。adaptive-shrink 會撞同一面牆，**未做**。
- ⇒ ✅ **M10 已做：boundary 接觸保持的 compaction**（pack 更密但不增 vBd，見上方 M10）。
  外加意外發現的**精度 bug**（虛假 fragment）才是最大 lever。**area_gap 仍是下個目標的主軸**
  （compaction 後 vCl 已大降，但 area dead space 只部分回收 — 見「下一步」）。

**對標**：組員 v5 = 1.7429（好 16.6%）、組員 v6/v7 portfolio ~1.62（**反超 ~10%**）。

**M6–M8 portfolio 層**（`optimizer_constructive.py`）：平行跑 **13 個 deterministic
profile**，用 **baseline-free proxy** 選最佳。profile 是 env 旋鈕變體，**兩大分歧軸**：
block boundary-aspect（高 LR → 低 vBd，violation-heavy case 贏）+ frame outline 形狀
（frame_tall 拿 13% 加權）。13 個：base / wire_hi / anc_lo / area_lean / aspect_hi(LR3.5) /
aspect_xhi(5.0) / asp_wire / aspect_v7(7.0) / aspect_v10(10.0) / asp7_wire / asp5_anclo /
**frame_tall**(aspects 0.67-0.33) / **frame_tight**(scales 1.0-1.2)。**加 profile 有
下檔保護**（proxy 幾乎完美 → 無用 profile 只是不被選、不傷分，只花 runtime）。
- proxy = `(area/Â + hpwl/hmin)·exp(2·vrel)`，Â=1.035·ΣblockArea，hmin=該 case 各
  profile 最小 hpwl（推導自 cost=0.5(area/A+hpwl/H)·exp(2·vrel)，vrel 精確、area
  baseline 可估、hpwl baseline 用 per-case 尺度）
- ⚠️ **vrel 必須用 shapely 算**（比照 harness）：C++ `count_group_fragments`
  用 1e-3 tol 把 MARGIN 間隙當接觸，與 shapely 在 ~34/100 案不一致 → 若用 C++
  vrel 選擇退到 1.61 區間。wrapper 內 `_proxy_metrics` 用 shapely 重算 → 命中天花板
- **oracle 天花板 = 1.5659，proxy 抓到 1.5659（完美）**：deterministic 無 SA
  限時噪音是關鍵（舊 SA portfolio proxy 只抓到天花板的一半）
- 子集邊際值（oracle）：7-prof 1.6057 → +4 aspect(11-prof) 1.5839 →
  +frame_tall 1.5709 → +frame_tight(13-prof) 1.5659。frame_wide/wwire 無用已棄
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 可退回單 base profile（1.7045）

---

**單一 profile 架構（每 profile 五階段，~0.16s/case；M1–M5）**：
1. **boundary-aspect dims**：LEFT/RIGHT-only aspect 2.50、TOP/BOTTOM-only 0.40
2. **MIB 形狀統一**（M4，`apply_safe_mib_dims`）：MIB group 有 fixed/preplaced
   master 且面積相容→全用 master 形狀；否則 movable 面積互相 ≤1%→全設 `sqrt(avg)`
   方形。保 1% area 硬約束 → feasibility 不變。**vMb 145→0**
3. **cluster 建構**：
   - 純 movable cluster → **複合 item**（3 ordering × 5 layout：h-row/v-col/方形
     shelf/寬 shelf/two-rows），選擇 key = `(fragments, boundary_bad, area, aspect)`
     字典序 — fragment/boundary 排 area **前面**（M4，-9.4%）
   - **mixed cluster（preplaced+movable）→ anchored**（M5）：movable 不進複合 item，
     而在 pack_in_frame **first-pass** 逐個貼 preplaced「牆」(`adjacent_candidates_
     for_block` 八向 abutment + boundary-exact)，不接觸 cluster 加 7000 penalty 保連通；
     失敗者退回 single。**vBd 359→307、vCl 260→208，-6.4%**
4. **定框 greedy packing**：試 4-5 個 outline frame（面積小優先），每 item 生
   boundary-aware 候選、評分（`bbox_area + 0.10·anchor + ww·WIRE·wire +
   BP_W·boundary_miss + BL`），挑最佳；layout_score 挑最佳 frame
   - **ww base ×2000**（M4，原 0.025-0.075 → 50/70/150）：bbox-area 最小化會散開
     連線，baseline 是 wire-driven。swept 最佳 ~2000-3000（**平坦盆地**非尖峰 →
     泛化安全；HPWL 本就是 cost 一半）
   - env 旋鈕：`ICCAD_BP_WEIGHT`(預設 30000) / `ICCAD_WIRE_MULT`(×base, 預設1) /
     `ICCAD_ANCHOR_W`(預設 0.10)
5. 3 個 repair nudge（boundary/group/edge-escape）— 在我們 layout 上多為 no-op

**演進（deterministic A/B）**：M1 singles 3.62 → M2 cluster 複合 2.3456 →
M3 +incremental wire 2.2515 → M4: +MIB 2.1673 → +cluster layout key 1.9638 →
+wire ×2000 1.8218 → M5: +anchored cluster 1.7045 → M6: +7-profile portfolio
1.6060 → M7: +4 aspect profile (11-prof) 1.5842 → M8: +frame_tall/tight (13-prof)
1.5659 → M9: +two-pass wire refinement 1.5375 → M10: 精度修正 + compaction 1.4528
→ M11: 迭代 compaction 1.4502 → M12: 40-profile 組合擴充 1.4371 → M13: narrow frame +
proxy _RH=1.4 修正 1.4349 → M14: post-placement HPWL push 1.4253 →
M15: HPWL push 擴大可移範圍 (boundary-axis slide) 1.4236 →
M16: HPWL push 加 same-size swap 1.4231 →
M17: WIRE_TIEBREAK pack-order 軸 + 2 profiles (46-prof) 1.4202 →
M18: WIRE_BFS pack-order 軸 + 3 profiles (49-prof) 1.4138 →
M19: BFS_PIN pin-anchored seed + 2 profiles (51-prof) 1.4105 →
M20: ORDER_SWAP pack-order hill-climb + 1 profile (52-prof) 1.4080 →
M21: ORDER_SWAP 組合掃描 + 3 profiles (55-prof) 1.3998 →
M22: OS16 移植無 PIN 排序 + 2 profiles (57-prof) 1.3987 →
**M23: ORDER_MOVE relocation 軸 + 1 profile (58-prof) 1.3983**（M23 portfolio
**-0.03%**；M4 起累計 **-38.0%**）。

**下一步（→ 繼續壓低天花板，當前 1.3983）**：proxy 已 = oracle ceiling（M13 起，
M18-M23 連續驗證 67/68 realize），selection 不再是瓶頸。pack-order 軸七連勝
（M17 WT -0.20%、M18 BFS -0.45%、M19 PIN -0.23%、M20 OS -0.18%、M21 OS 組合 -0.58%、
M22 OS16 移植 -0.08%、M23 OM -0.03%）但 **OS K 軸（M22）與 OM 軸（M23）皆已收割**
→ **剩餘新 C++ 行為軸**。按預估 ROI：
1. ~~迭代 compaction~~ ✅ M11、~~profile 擴充~~ ✅ M12、~~HPWL push~~ ✅ M14/M15/M16、
   ~~WIRE_TIEBREAK 軸~~ ✅ M17、~~WT 組合 + WIRE_BFS 軸~~ ✅ M18、~~BFS 組合 + BFS_PIN~~
   ✅ M19、~~ORDER_SWAP 軸~~ ✅ M20、~~OS 組合掃~~ ✅ M21、~~OS K 放大（24/32）+ OS16
   移植~~ ✅ M22（OS24/32 驗證有效但 runtime 回退）、~~ORDER_MOVE 軸 + cluster pack-order
   位置~~ ✅ M23（om8 ship；om16 候補；CLUSTER_ORD 死）
2. ~~cluster-rigid pack/slide~~ ❌ 兩次失敗；~~violating boundary 修復~~ ❌ M16（0 可修）；
   ~~profile knob 軸~~ ❌ M16；~~per-frame csc~~ ❌ M17；~~NO_COMPACT profile~~ ❌；
   ~~WT knob 組合~~ ❌ M18（7 個 ≤0.03%）；~~BFS knob 組合~~ ❌ M19（8 個僅 1 過門檻）；
   ~~BFS_NORM（attach/√area）~~ ❌ M20（4 個 ≤0.026%）；~~PIN 補掃（tight/narrow）~~
   ❌ M20（≤0.030%）；~~OS+WT+tall / OS16+tall / OS12+PIN~~ ❌ M21（被入選三者蓋過）；
   ~~CLUSTER_ORD~~ ❌ M23（兩排序全 0.000%）；~~OM+tight~~ ❌ M23（0.000%）
3. **🔑 剩餘新 C++ 行為軸**：refinement pair-relocation（placement 層 — 在 refinement
   迴圈內把放壞的 block 拔出重插別的 slot，M19 提過未做）、compaction 方向偏好 /
   pack over-spread 軸先（dbg_area：多數 case 寬度過寬）、pack 向 connectivity 重心。
4. **runtime 候補（+0.23% 現成，等規則確認）**：os32_pin（case 71 1.3187 + 89 1.8093）
   + os24_pin（case 66 1.3989）+ **om16_bfs_wt_wire（M23：case 96 1.3160 + 66 1.3951
   + 91/42/53，live 驗證 59-prof = 1.3968、20/20 realize）** → 若官方 RuntimeFactor
   （cross-submission median）確認寬鬆可直接加回。**懲罰比公式（M23 確立）：兩配置
   factor 比 = (t1/t2)^0.3 與 median 無關** → 加重 profile 的決策只看「增益是否付得起
   確定性溢價」。
5. **case 66 半收割**（1.4378，om16 可到 1.3951）；**case 89**（1.8061 = M23 新低，
   os32 1.8093 已被超越）；**case 96**（1.3336，om16 可到 1.3160）；85（1.6255）。
次大殘留：硬 case（89 1.8061、85 1.6255）多為 preplaced boundary 撐壞 outline。
violation 殘留結構（M16 量測）：202 violating boundary block = 123 cluster member + 45
preplaced + 34 blocked single → **後處理不可修，只能靠 packing 階段擺對**。

**⚠️ 已驗證 BP_WEIGHT 不是 lever**：30000→1M 完全無變化 → boundary violation 不是
「penalty 太低被 area 蓋過」，而是「無可行 bp=0 位置」或「frame 邊 ≠ bbox 邊」。
compaction/penalty 路線對 cluster boundary 無效。

⚠️ 學到：**「試更多 frame」會退步**（all-frames 2.42）— layout_score 的 150000·bv
權重在大候選池中 overshoot（挑了低 violation 但 area 爆掉的 outline）。4-5 frame 最佳。

---

### 舊最佳：Total Score = **3.0625** (Portfolio 8-profile, 2026-06-01)

Portfolio wrapper（code 已刪，僅留紀錄）— **8** profile 並行（gnn / connectivity /
area_desc / area_asc / pin_centroid / degree_desc / degree_asc / **high_boundary**）
每個吃 full 8s SA，contest-shape proxy 挑最佳。
- Avg Cost 2.3944（vs 2.6548 baseline, -9.8%）
- 100/100 feasible
- Wall time 9.13s（12 physical cores, 8 subprocess 並行）
- `high_boundary` 機制：connectivity perm + **W_BOUNDARY=100** (10× default)
  via env var `ICCAD_W_BOUNDARY` — C++ `main()` 已加 env var override 支援
- Winner 分佈 (100 cases)：**high_boundary 18 (#1)**, gnn 14, pin_centroid 14,
  connectivity 13, degree_asc 12, degree_desc 11, area_desc 10, area_asc 8
- **high_boundary 主導大 n cases**：n=100-109 拿 4/10, n=110-119 拿 3/10,
  n=120 拿 1/1。Violation-heavy top-10 worst 拿 3 個（case 99/95/63）。
- 確認 hypothesis：W_BOUNDARY 10 → 100 amplify soft boundary gradient，
  幫助 SA 在大 n 高 violation cases escape 局部極小。

### 前一版：Total Score = **3.1082** (Portfolio 7-profile, 2026-06-01)
7 profile（無 high_boundary）, Avg Cost 2.4260, Wall time 9.35s.
⚠️ run-to-run 變異 ±2%：同 code 跑過 3.1724 也跑過 3.1082。

### 歷史最佳（單跑）：Total Score = **3.2708** (新評分)

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
| 2026-05-31 + boundary aspect (2.50/0.40, teammate v5) | **3.4255 退步**，已 revert |
| 2026-05-31 portfolio (4 profile 並行, contest-shape proxy) | 3.1584 ← -5% vs 3.3258 |
| 2026-06-01 portfolio (7 profile：+pin_centroid/degree_desc/degree_asc) | 3.1082 ← -1.6% vs 4-profile |
| **2026-06-01 portfolio (8 profile：+high_boundary W_BOUNDARY=100)** | **3.0625** ← **新最佳, -1.5% vs 7-profile, -9.8% vs baseline** |
| 2026-06-02 portfolio (10 profile：+low_viol/high_viol, W_VIOL ×0.5/×2.0) | 3.0859 ← **退步 +1.0%, 已 revert**（同機 clean A/B: 8-prof 3.0554 vs 10-prof 3.0859）|
| 2026-06-02 proxy near-tie min-viol tie-break (margin 0.02) | **離線固定輸出 A/B 3.1288 → 3.0979 (-1.0%)**, 已 ship；live run 3.1040（被 ±2-3% 限時噪音蓋過）|
| 2026-06-02 oracle-selector 天花板 (8-profile) | 3.0335 ← **完美選擇也破不了 3.00**，selection 已盡 |
| **2026-06-04 constructive placer M1 (singles)** | 3.62 ← 架構可行但缺 cluster/wire |
| **2026-06-04 constructive M2 (cluster 複合 item)** | 2.3456 ← **-35% vs M1, 破 portfolio 3.05** |
| **2026-06-04 constructive M3 (+nudges +incremental wire)** | **2.2515** ← -26% vs portfolio, 0.12s/case |
| 2026-06-04 constructive M4a (+MIB 統一, vMb 145→0) | 2.1673 ← -3.7% |
| 2026-06-04 constructive M4b (+cluster layout key: frag/boundary 排 area 前) | 1.9638 ← -9.4%, vBd 528→357 |
| 2026-06-04 constructive M4c (+wire weight ×2000, anchor 0.1) | 1.8218 ← -7.2%, 距組員 1.74 僅 4.7% |
| 2026-06-05 constructive M5 (+anchored cluster first-pass) | 1.7045 ← -6.4%, 反超組員 v5 (1.7429) |
| 2026-06-05 constructive M6 (+7-profile portfolio + baseline-free proxy) | 1.6060 ← -5.8%, 反超組員 v6/v7 (~1.62); oracle 天花板 1.6057 |
| 2026-06-05 constructive M7 (+4 aspect profile → 11-prof) | 1.5842 ← -1.4%, oracle 天花板 1.5839 (proxy 抓滿) |
| 2026-06-05 constructive M8 (+frame_tall/tight → 13-prof) | 1.5659 ← -1.2%, frame outline 新分歧軸; oracle 1.5659 |
| 2026-06-06 constructive M9 (+two-pass wire refinement, iters=12) | 1.5375 ← -1.8%, 攻 HPWL gap; 單 base 1.7045→1.658; runtime 1.36s/case |
| 2026-06-06 +frame_fine profile (14-prof, tighter outline 給 area-dominated case) | 1.5362 ← -0.08% (marginal); area frame 路線枯竭, 見 area 分析 |
| 2026-06-06 constructive M10a (輸出精度 %.10f→%.17g, 消虛假 cluster fragment) | 單 base 1.658→1.5532 (-6.3%) ← **最大單一 lever, 既有 pipeline 一直在漏** |
| **2026-06-06 constructive M10b (+boundary 保持 compaction, csc 選擇)** | **1.4528** ← portfolio -5.4%; 單 base 1.5532→1.5335; 100/100 feasible |
| 2026-06-07 constructive M11 (迭代 compaction: pack_x→pack_y→pack_x…, COMPACT_ITERS=8) | 1.4502 ← -0.18%; 單 base 1.5335→1.5306; 收斂 1 輪 |
| 2026-06-07 cluster-rigid compaction (整 cluster 當剛體滑) | 1.5464 ← **退步, revert**（1.5306→1.5464, vCl 62→67） |
| 2026-06-07 constructive M12 (40-profile 組合擴充: +26 combo profiles) | 1.4371 ← -0.87% vs M11; 突破 < 1.43 目標; 100/100 feasible; 4.0s/case |
| 2026-06-07 M13a (+4 narrow-frame profile, aspect 0.55-0.28, 攻水平 dead space) | 1.4369 ← 贏 case 98/87 真值, 但 proxy(_RH=1.0) 選不出 → 幾乎沒動 |
| 2026-06-07 constructive M13b (proxy _RH 1.0→1.4: hmin/hbase 補償) | 1.4349 ← -0.15% vs M12; proxy = oracle ceiling 1.4349; 100/100 feasible; 4.59s/case |
| 2026-06-07 preplaced-aligned frame (攻 case 89) | **失敗, revert** ← greedy pack 不下 tighter width; 贏的案全被現有 profile 蓋過 (零貢獻) |
| 2026-06-07 constructive M14 (post-placement HPWL push, free single → L1-median, default-on) | 1.4253 ← -0.67% vs M13; 攻 hgap (最大 lever 0.412); downside-free (area/bv/gf/mib 不變); 單 base 1.5306→1.5219; 100/100 feasible; 3.85s/case |
| 2026-06-08 cluster-rigid slide in HPWL push (dbg_hpwl_push ENABLE_CLUSTER) | **失敗, 不實作** ← cluster 無 slack (100 案僅 1 個能動) + FP 平移破壞精確 abutment → shapely 虛假 fragment (M10 陷阱); 淨 +0.004% 且 1 退步 |
| 2026-06-08 constructive M15 (HPWL push 擴大可移範圍: +boundary-axis slide, default-on) | 1.4236 ← -0.12% vs M14; 接續攻 hgap; boundary block 滑 free 軸 (LR→y, TB→x) 保 edge-contact; downside-free (單 base 1.5219→1.5202, vBd/vCl/vMb 270/62/0 不變); 0/100 退步; 100/100 feasible; 4.40s/case |
| 2026-06-10 violating boundary single 修復 (dbg_hpwl_push ENABLE_VIOLATING) | **死路, 不實作** ← 推到邊修 bv: 真值 0 個修好。dbg_vio_stats: 202 violating = 123 cluster + 45 preplaced + 34 single **全 BLOCKED** → 後處理不可修 |
| 2026-06-10 profile knob 軸掃描 (WIRE_MULT 4/6, LR3.5+W4, ANCHOR 0.30, ultra-narrow) | **枯竭, 不加** ← 最佳 W4.0 oracle-min 僅 +0.063%, 其餘 ≤0.02% (profile_vs_portfolio.py) |
| 2026-06-10 constructive M16 (HPWL push 加 same-size swap, default-on) | 1.4231 ← -0.04% vs M15; 同 (w,h,boundary-code) 的 non-pre non-cluster pair 交換位置 (嚴格降 HPWL); 幾何 multiset 不變 → downside-free (單 base 1.5202→1.5197, vBd/vCl/vMb 270/62/0 不變); 0/100 退步; 100/100 feasible; 3.99s/case |
| 2026-06-10 per-frame compaction + csc 重估 frame (M17 實驗) | **死路, code 已移除** ← default-on 單 base 1.5197→1.5293 (csc 固定 hw 跨 outline 失準, 拿 vCl 換 vBd); 當 profile oracle-min +0.008%。跨 layout 選擇是 wrapper shapely proxy 的工作 |
| 2026-06-10 constructive M17 (WIRE_TIEBREAK pack-order 軸 + wtb_wire/wtb_tall_wire, 46-prof) | 1.4202 ← -0.20% vs M16; bscore 第一鍵不動、同類內 total_wire 大者先放; wtb_tall_wire 破解 case 79 (1.706→1.597, 最大 hgap 單案) + 89/53/16; wtb_wire 拿最高權重 case 98 (1.4502→1.4413) + 63; 100/100 feasible; 3.86s/case |
| 2026-06-10 WT knob 組合掃描 (anc_lo/tight/asp7/W2+anc/tight+W2/asp7+W2, M18 實驗) | **WT 軸枯竭** ← 7 個組合 ≤0.03%, 只有 WT+tall+anc +0.089% (case 87 獨有) 過門檻已加 |
| 2026-06-10 constructive M18 (WIRE_BFS pack-order 軸 + bfs_wt_wire/bfs_tall_wire/wtb_tall_anc, 49-prof) | 1.4138 ← -0.45% vs M17 (M14 以來最大單 session); bscore 類內 BFS-connectivity greedy attachment 重排 (與已排 items+preplaced 連接權重最大者先放, tie 保基序); bfs_wt_wire +0.316% 拿 case 98 (1.4413→1.4221)+95/91/50; bfs_tall_wire +0.251% 零重疊拿 case 79 (1.597→1.525)+86/74/97/89; 16/16 預估贏案全 realize (proxy=oracle); 100/100 feasible; 4.51s/case |
| 2026-06-10 BFS knob 組合掃尾 (asp5/asp7/WT+tall/narrow+anc/WT+narrow/WT+asp5/tight/anc, M19 實驗) | **BFS knob 軸掃尾完成** ← 8 個只有 tight+W2 +0.061% (case 91) 過門檻; anc+W2 +0.063% 被 PIN 蓋過; 其餘 ≤0.025% |
| 2026-06-10 constructive M19 (BFS_PIN pin-anchored seed + bfs_pin_wt_wire/bfs_tight_wire, 51-prof) | 1.4105 ← -0.23% vs M18; BFS seed attachment 加 p2b pin 權重 (pins 是固定 anchor, 同 preplaced); bfs_pin_wt_wire +0.269% 再殺 case 95 (1.2995→1.2767)+94 (1.3656→1.3411)+64; bfs_tight_wire +0.061% 拿 case 91+小案; 13/14 預估贏案 realize (miss 僅 case 53 差 0.0009/權重 0); 100/100 feasible; 4.48s/case |
| 2026-06-10 BFS_NORM (attach/√area) + PIN 補掃 (tight/narrow, M20 實驗) | **兩軸死路** ← NORM 4 組合 ≤0.026%、PIN 補掃 ≤0.030%, 全部主贏案是 case 66 (~1.389 可達但單案 ~0.03% 不過門檻) |
| 2026-06-10 constructive M20 (ORDER_SWAP pack-order hill-climb + os_pin_wt_wire, 52-prof) | 1.4080 ← -0.18% vs M19; 每 frame refinement 前對 top-8 total_wire items 28 對做 pack-order 互換 hill-climb (pack-once 比較, layout_score 嚴格改善才收, items_base 隔離); os_pin_wt_wire +0.252% 再殺 case 94 (1.3411→1.3128)+98 (1.4221→1.4118)+79 (1.5249→1.5135)+50/28/9; OS8 單獨僅 +0.030% (低權重廣效) → swap 需好的起始 order; 12/12 realize; 100/100 feasible; 5.10s/case |
| 2026-06-11 OS 組合掃尾 (OS8+WT+tall / OS12+PIN / OS16+BFS+tall, M21 實驗) | **不加** ← OS12+PIN +0.157% (case 89 1.8106 最深) 與 OS16+tall +0.056% 全被入選三 profile 蓋過 (殘留 ≤0.017%); OS8+WT+tall +0.016% |
| 2026-06-11 constructive M21 (ORDER_SWAP 組合掃描 + os16_pin_wt_wire/os_bfs_wt_wire/os_bfs_tight_wire, 55-prof) | 1.3998 ← -0.58% vs M20, 突破 < 1.40; 無 C++ 改動純 portfolio 層; os16_pin_wt_wire +0.460% (K=16=120 對 swap 池) 再殺 case 98 (1.4118→1.3841)+79 (1.5135→1.4219)+82+89 首次鬆動 (1.8273→1.8155); os_bfs_wt_wire +0.262% 拿 86 (1.3775→1.3347)+95+97; os_bfs_tight_wire +0.221% 首殺硬 case 85 (1.6606→1.6255)+42+40; 三者贏案近零重疊; 12/12 realize (掃描預估 1.3999 vs 實際 1.3998); 100/100 feasible; 8.36s/case |
| 2026-06-11 OS K 放大掃描 (OS24/OS32 on PIN+WT, M22 實驗) | **驗證有效但 runtime 回退** ← 四個全加 59-prof live = **1.3979**（掃描精確吻合, 100/100 feasible）但 avg runtime 8.4→**21.9s/case**（OS24/32 在 n=120 cpu 32s/58s）；官方 RuntimeFactor=max(0.7,R^0.3) 分母 cross-submission median 未知（組員 ~11s 唯一參考）→ 0.04% 分數差扛 20-35% 懲罰風險不划算 → 回退 os32_pin（71 1.3187+89 1.8093）/os24_pin（**66 1.3989 唯一解**），候補留 code 註解；**K=16 飽和證明：98/79/82 在 OS24/32 贏案全絕跡** |
| 2026-06-11 constructive M22 (OS16 移植兩個無 PIN 排序 + os16_bfs_wt_wire/os16_bfs_tight_wire, 57-prof) | 1.3987 ← -0.08% vs M21; os16_bfs_wt_wire +0.056% 磨深 86 (1.3347→1.3255)+82 (1.4679→1.4565)+50 (1.2920→1.2419); os16_bfs_tight_wire +0.057% 首殺 case 62 (1.6214→1.5248)+55+51+66 半收 (1.4450→1.4378); 8/8 realize; 100/100 feasible; 8.79s/case; wrapper timeout 55→120s |
| 2026-06-12 M23 掃描: OM16+BFS+WT (+0.148%) / OM8+OS16 (+0.070%) / OM16+PIN (+0.040%) / OM12 (+0.012%) / OM8+tight、CLUSTER_ORD×2 (0.000%) | **om16_bfs_wt 驗證有效但 runtime 回退** ← 59-prof live = 1.3968 (20/20 realize, case 96 首殺 1.3160 + 66 收割 1.3951) 但 avg 13.51s/case; (13.51/8.8)^0.3 = +13.8% 懲罰比 (與 median 無關) 換 +0.14% 不划算 → 候補化 (與 os24/os32 並列, 合計 +0.23% 現成); **CLUSTER_ORD 軸死路** (兩排序 0.000%); OM12 非線性丟失 96/66 → K 無便宜折衷 |
| **2026-06-12 constructive M23 (ORDER_MOVE relocation 軸 + om8_pin_wt_wire, 58-prof)** | **1.3983** ← **新最佳, -0.03% vs M22; ORDER_MOVE=K 拔出重插 jump move (swap 之外的新結構移動); om8_pin_wt_wire +0.041% — case 89 歷來最深 (1.8155→1.8061, 勝 os32 候補 1.8093)+57 (1.4011→1.3689)+26/35/33/2; 6/6 realize; 100/100 feasible; 9.49s/case 安全帶內** |
| 【外部驗證】組員 my_optimizer.py 餵我們 evaluator | 1.7429 ← 確認架構可移植 |
| 2026-05-31 oracle shape only (sanity)  | 3.4199 ← **shape ML 死** (改善 0.3%) |
| 2026-05-31 oracle shape + oracle perm | 3.3672 ← 鎖死 shape 反害 SA |
| 2026-05-26 v2 supervised MSE on fp_sol (2000 sample, < 3h) | **失敗** — pos_mse 震盪、unsup_cost 47M，.pth 已棄 |
| 2026-05-27 v3 sanity (120 sample, 30 batches) | rank_acc 0.53 → 0.58，訊號弱 — 待 oracle 實驗決定 |
| **2026-05-31 oracle perm + SA (上限實驗)** | **3.2673** ← BL packer 是天花板，v3 ML 放棄 |
| **【外部參考】組員 v6/v7 portfolio (legit, 無 label)** | **~1.62** ← 真正可達目標 |
| 【外部參考】組員 v9 oracle (讀 label) | 1.0322 ← hidden test 不適用 |

---

## 這個階段想解決的問題（constructive M23 後，當前 1.3983）

> 舊 SA 範式的瓶頸（slack=0 boundary、SA 收斂、bbox shrinking）已隨架構換成
> constructive placer 而作廢。以下是**當前** placer 的瓶頸，依 leverage 排序。
> ⚠️ **gap 分解（讀 results.json 加權）：weighted hgap ≫ agap 0.23 ≫ vrel 0.038。
> HPWL gap 是壓倒性主 lever**（cost=(1+0.5(hgap+agap))exp(2vrel)）。

### A. HPWL gap 是最大 cost lever（最高 leverage，M14-M23 主軸）
- M14 free-single push (-0.67%) + M15 boundary-axis slide (-0.12%) + M16 same-size swap
  (-0.04%)：**post-placement 微調已收割完**；pack-order 軸連七次得分（M17 WT -0.20%、
  M18 BFS -0.45%、M19 PIN -0.23%、M20 ORDER_SWAP -0.18%、M21 OS 組合 -0.58%、
  M22 OS16 移植 -0.08%、M23 ORDER_MOVE -0.03%）證明 hgap 殘留要靠 **packing 結構**收
- ❌ 封死：cluster-rigid（無 slack + FP 破壞 abutment）；violating boundary 修復（M16 實測
  0 可修）；per-frame csc（M17）；WT knob 組合（M18）；BFS knob 組合（M19）；
  BFS_NORM、PIN 補掃（M20，全 ≤0.03%）；CLUSTER_ORD（M23，0.000%）— **knob/輕變體
  空間徹底掃完**，新增益全來自新 C++ 行為與其組合
- case 79：M17 破解（1.706→1.597）→ M18 再破（→1.525）→ M20 再降（→1.5135）→
  M21 OS16 又破（→**1.4219**）；case 98 連四殺至 **1.3841**；case 89 M21 首鬆動 →
  M23 OM8 歷來最深（→**1.8061**）— wire-driven 聚集 + order jump move 持續有效。
  **⚠️ M22 證明 OS K 軸枯竭**（K=16 飽和於高權重案、K>16 runtime 不划算）；
  **M23 證明 OM 軸同樣短**（om8 ship +0.041%；om16 +0.148% 驗證有效但 runtime 候補化；
  OM12 非線性丟失主贏案 → K 無折衷；OM×tight 無效）。**order 層 jump move（swap+move）
  全收割完** → 下一步：placement 層 pair-relocation、compaction 方向、connectivity 重心。

### B. area_gap dead space（次大 uniform 缺口）
- M10/M11 compaction 後 density 仍 >1.1（原圖 1.035）→ 還有 void 可擠
- compaction 逐 block pack，靠 csc 拒絕 fragment 的候選 → 較保守，未榨乾
- agap outlier：case 79、99 等 tighter frame pack 不下 → 退到大/寬 frame；compaction 後可
  **重估** frame 選擇（大 frame 易 pack + compaction 擠掉 void → 可能勝過原本選的 tight frame）

### C. 硬 case：preplaced boundary block 撐壞 outline
- case 89 (hgap 0.751 + vBd 7)、85 (vBd 10)：preplaced 位置固定，bbox 邊到不了它
- ⚠️ **frame 偏好「不超出 preplaced 外緣」已試並失敗（M13）**：greedy packer **pack 不下**
  width 釘在 preplaced 外緣的 tighter frame（case 89 完全不動）。結構性無解，見 M13 段。

### D. proxy selector / profile 多樣性（M13 後 proxy = oracle ceiling）
- ✅ **M13 已修好 proxy**：_RH 1.0→1.4（補償 hmin/hbase≈1.3-1.4）→ proxy **完美命中
  oracle 天花板 1.4349**。selection 不再是瓶頸。
- 加 profile 現在**全額 realize**（M12 加 profile 半realize 是因為 _RH=1.0 選不準）→
  oracle ceiling 本身才是新天花板。下一步壓 ceiling 要靠**新 layout 多樣性**（新 profile 軸
  或 placer 改進降低每案最佳 cost），不是 selection。
- proxy 用 **shapely vrel**（wrapper `_proxy_metrics`）。掃 _RH/proxy 參數用 `rh_sweep.py`。

---

## 預期目標

> 基準線：constructive portfolio **1.3983**（M23）。對標：組員 legit portfolio
> ~1.62（**已反超 ~13.7%**）、組員 oracle 1.0322（讀 label，hidden test 不適用）、
> fp_sol verbatim 1.1079（理論重建上限）。確定性 → 可精確 A/B，無 SA 限時噪音。

### 已達成
- ✅ Total Score < 3.00 / < 2.00 / < 1.60 / < 1.43 / < 1.42 / < 1.41 / **< 1.40**（當前 1.3983）
- ✅ **反超組員所有 legit 版本**（v5 1.7429、v6/v7 portfolio ~1.62）
- ✅ baseline-free proxy ≈ oracle 天花板（無 label leak，hidden test 可用）
- ✅ M10 精度修正（消虛假 fragment）+ boundary 保持 compaction（攻 area_gap）
- ✅ M11 迭代 compaction（1.4528→1.4502）
- ✅ M12 40-profile 組合擴充（1.4502→1.4371，< 1.43 目標達成）
- ✅ M13 narrow-frame profile + proxy _RH=1.4（1.4371→1.4349，proxy = oracle ceiling）
- ✅ M14 post-placement HPWL push (free single)（1.4349→1.4253，攻 hgap，downside-free default-on）
- ✅ M15 HPWL push 擴大可移範圍 (boundary-axis slide)（1.4253→1.4236，downside-free default-on）
- ✅ M16 HPWL push 加 same-size swap（1.4236→1.4231，downside-free default-on）+
  三個死路驗證（violating 修復 / profile knob 軸 / case 79 cheap trick，省下後續 session 重試）
- ✅ M17 WIRE_TIEBREAK pack-order 軸 + wtb_wire/wtb_tall_wire profiles（1.4231→1.4202，
  破解 case 79 + 拿下 case 98）+ per-frame csc 死路驗證
- ✅ M18 WIRE_BFS pack-order 軸 + bfs_wt_wire/bfs_tall_wire/wtb_tall_anc profiles
  （1.4202→1.4138，case 98/95/79/87 全下殺，16/16 realize）+ WT knob 軸枯竭驗證
- ✅ M19 BFS_PIN pin-anchored seed + bfs_pin_wt_wire/bfs_tight_wire profiles
  （1.4138→1.4105，case 95/94/91/64 下殺，13/14 realize）+ BFS knob 軸掃尾
- ✅ M20 ORDER_SWAP pack-order pair-swap hill-climb + os_pin_wt_wire profile
  （1.4105→1.4080，case 94/98/79/50 再殺，12/12 realize）+ NORM/PIN 補掃死路驗證
- ✅ M21 ORDER_SWAP 組合掃描 + os16_pin_wt_wire/os_bfs_wt_wire/os_bfs_tight_wire
  （1.4080→**1.3998**，< 1.40 達成；case 98/79/86/95/85/82/89 全下殺，12/12 realize）
- ✅ M22 OS K 放大掃描收尾 + os16_bfs_wt_wire/os16_bfs_tight_wire（1.3998→**1.3987**，
  case 62 首殺 + 86/82/50/55/51，8/8 realize）+ K 軸飽和證明 + runtime 風險決策
  （os24/os32 驗證 1.3979 但 21.9s 回退，候補 +0.08%）
- ✅ M23 ORDER_MOVE relocation 軸 + om8_pin_wt_wire（1.3987→**1.3983**，case 89 歷來
  最深 1.8061 + 57/26，6/6 realize；9.49s/case）+ om16 候補化（+0.148% 驗證有效，
  live 1.3968/20/20 realize，但 13.51s/case 懲罰比 +13.8% 不划算）+ CLUSTER_ORD
  死路驗證 + 懲罰比公式確立（factor 比 = (t1/t2)^0.3 與 median 無關）

### 短期（當前目標）
- ~~**目標 1**：Total Score < 1.43~~ ✅ M12 完成（1.4371，-0.87% vs M11）
- ~~**目標 1b**：proxy 命中 oracle ceiling~~ ✅ M13 完成（_RH=1.4 → 1.4349 = ceiling）
- ~~**目標 2**：Total Score < 1.40~~ ✅ M21 完成（1.3998，-0.58% vs M20）
- **目標 2b (當前)**：Total Score < 1.39 — 還差 ~0.6%。**OS K 軸（M22）與 OM 軸（M23）
  皆已收割** → 需 placement 層新行為。
  - 候選：refinement pair-relocation（placement 層拔出重插）、compaction 方向偏好 /
    pack over-spread 軸先、pack 向 connectivity 重心
  - **runtime 候補 +0.23% 現成**：os32_pin + os24_pin（1.3979）+ om16_bfs_wt_wire
    （M23 驗證 1.3968、case 96 1.3160/66 1.3951）等官方 RuntimeFactor 規則確認寬鬆即加回
  - case 66 半收（1.4378，om16 可到 1.3951）、89（1.8061 = M23 新低）、96（1.3336，
    om16 可到 1.3160）、85（1.6255）
  - ⚠️ cluster-rigid slide 路線已封死（M11+M15 兩次失敗：無 slack + FP 破壞 abutment）；
    ⚠️ CLUSTER_ORD 已死（M23）

### 中期（4–6 個迭代）
- **目標 3**：Total Score < 1.35 — 需 placer 結構升級（壓 oracle ceiling，非 selection）
- **目標 4**：硬 case（preplaced boundary 撐壞 outline）— ⚠️ frame 偏好策略已試失敗（M13），
  需 packer 能 pack tight（greedy packer 升級）才可能解
- **目標 5**：profile 軸擴充（~~cluster ordering 變體~~ ❌ M23 CLUSTER_ORD 死路），
  proxy 完美 → 全額 realize（不再半realize）

### 長期（逼近重建上限 ~1.1）
- **目標 7**：從「optimization（找好解）」轉向「reconstruction（還原原圖）」——
  真正的天花板在 ~1.03-1.11，需用 connectivity + constraints 反推 baseline 佈局，
  而非只壓 area/hpwl/violation。見頂部「🚨 範式轉移」段落。
- **目標 8**：把 supervised ML（structural ranking）整合成 placer 的 perm/起點 hint
  （但須先確認 placer 不再是天花板 — 舊 oracle-perm 實驗顯示 SA placer 是；
  constructive placer 的 oracle-perm 上限**未重測**，值得一試）

---

## 未來發展方向

> 全部以 constructive placer（`constructive.cpp`）為基礎。舊 SA 方向（slack push、
> W_BOUNDARY ramp、SA restart、skyline incremental）已作廢。

### 1. HPWL push 進化（攻 hgap = 最大 cost lever）— **微調路線已收割完（M14-M16）**
- ✅ **M14 已做**：post-placement `hpwl_push()` 滑 free single → connectivity L1-median
  進 bbox 內 void（downside-free，default-on）。-0.67%。
- ✅ **M15 已做**：擴到 **boundary-axis slide** — LEFT/RIGHT block 只滑 y、TOP/BOTTOM 只滑 x
  （constrained 軸不動 → edge-contact 保住 → bv 不變）。slide_x/slide_y lambda 共用。-0.12%。
  env `ICCAD_NO_BND_PUSH=1` 退回 M14。
- ✅ **M16 已做**：**same-size swap** — 同 (w,h,boundary code) 的 non-pre non-cluster pair
  交換位置（嚴格降 HPWL）。幾何 multiset 不變 → downside-free。-0.04%。`ICCAD_NO_SWAP=1`。
- ❌ **cluster-rigid slide 已封死**：dbg_hpwl_push.py 的 `ENABLE_CLUSTER` 驗證 → cluster 無
  slack（100 案僅 1 個能動）＋ FP 剛體平移破壞精確 abutment → shapely 虛假 fragment（M10 陷阱）。
  淨 +0.004% 且 1 退步。**任何移動 cluster 成員的後處理都會撞此精度牆**。
- ❌ **violating boundary 修復已封死**（M16 實測，dbg_hpwl_push.py `ENABLE_VIOLATING` +
  `dbg_vio_stats.py`）：202 violating boundary block = 123 cluster member + 45 preplaced +
  34 single 全 BLOCKED → 真值 0 個修好。residual vBd **只能靠 packing 階段擺對**。
- 收益遞減 -0.67% → -0.12% → -0.04% → **本方向結案**，轉攻 packing 結構（見 2/4）。

### 1b. compaction 進化（攻 area_gap，次高 ROI）
- ✅ M11 迭代 pack（pack_x→pack_y→…）已做（收斂 1 輪）。
- **cluster-rigid compaction** ❌ 已試失敗（1.5306→1.5464，revert，root cause 未確認）。
- **起點/順序**：先 pack over-spread 軸（`dbg_area.py` 顯示多數 case 寬度過寬）。

### 2. frame 選擇與 compaction 協同（**現最高 ROI 候選**）
- compaction 後重估 frame：大 frame 易 pack + compaction 擠 void，可能勝過原本選的
  tight frame（特別是 agap outlier 79/99）。可在 solve() frame loop 內對每 frame
  compaction 後再比 csc（注意 overfit 風險，見「per-frame 退步」教訓 — 該教訓針對
  layout_score；csc 池選擇未試過）。

### 3. 硬 case 處理（preplaced boundary 撐壞 outline）
- case 89/85：frame 偏好「不超出 preplaced 外緣」，或對這類 case 特化 outline。
- 用 `dbg_boundary.py` 分類違反（single/cluster/preplaced × blocked/free）；
  全 portfolio 統計用 `dbg_vio_stats.py`（M16 新工具）。

### 4. profile 軸擴充 — ⚠️ **env knob 變體已枯竭（M16/M18/M19/M20/M23 掃描）**
- ❌ WIRE_MULT 4/6、LR+W 組合、ANCHOR 0.30、ultra-narrow frame：oracle-min 全 ≤+0.063%；
  ❌ WT knob 組合（M18）；❌ BFS knob 組合（M19）；❌ BFS_NORM（attach/√area，M20 掃 4 個
  ≤0.026%）；❌ PIN 補掃（M20，≤0.030%）；❌ CLUSTER_ORD（M23，0.000%）。
- ✅ **M17-M23 驗證：有用的新軸 = 新 C++ 行為與其組合**（WIRE_TIEBREAK -0.20%、WIRE_BFS
  -0.45%、BFS_PIN -0.23%、ORDER_SWAP -0.18%、OS 組合掃 K=16 -0.58%、OS16 移植 -0.08%、
  ORDER_MOVE -0.03%）。❌ OS K>16 已枯竭（M22）；❌ OM K×排序組合掃完（M23：om16 候補、
  OM12 非線性、OM×tight 死）。**order 層收割完** → 下一個候選：refinement
  pair-relocation（placement 層）、compaction 方向偏好、pack 向 connectivity 重心。

### 5. 重建方向（逼近 ~1.1 上限，研究型）
- 當前仍是 optimization（壓 area/hpwl/violation）。真天花板需 **reconstruction**：
  用 b2b/p2b connectivity + constraints 反推「原圖怎麼擺」。
- 重測 **constructive placer 的 oracle-perm 上限**（舊上限實驗是對 SA placer 做的，
  結論 3.27 是 SA 的天花板；constructive 可能不同）→ 決定 ML ranking 值不值得。

### 6. 精度 / 數值（已部分做）
- ✅ 輸出 %.17g（M10，消虛假 fragment）。
- 注意：任何新加的、會被 shapely 評分的幾何輸出都要保持精確 abutment + %.17g。

---

## 環境

- **主程式**: `constructive.cpp` (C++ placer) + `optimizer_constructive.py` (portfolio wrapper)
  - 舊 SA：`optimizer_claude.cpp` / `.py` 仍在，僅作 constructive 失敗時的 fallback
- **Conda env**: `C:\Users\Nordra\.conda\envs\iccadv\python.exe`
- **Compiler**: `C:\msys64\ucrt64\bin\g++.exe`

### 編譯
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet"
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive.exe constructive.cpp
# 注意：Bash 工具寫 .exe 會失敗，用 PowerShell 編譯
```

### 評估（官方 portfolio，~3 分鐘；constructive 確定性、快）
```powershell
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_constructive.py 2>&1 | Select-Object -Last 12
```

### 快速 A/B（單 profile，~70 秒，乾淨確定性）
```powershell
# analyze_constructive.py 直跑 constructive.exe（base profile），重算 vBd/vCl/vMb + Total Score
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_constructive.py 2>&1 | Select-Object -First 12
# 關 compaction：$env:ICCAD_NO_COMPACT="1"；退單 base profile：見 ICCAD_CONSTRUCTIVE_SINGLE
```

### 分析工具
```powershell
& "...python.exe" dbg_area.py            # area_gap 分解 (density ours vs baseline)
& "...python.exe" dbg_compact.py         # compaction 原型 (orig vs compacted Total Score)
& "...python.exe" dbg_compact_cmp.py 99  # 單案 Py-原型 vs C++-binary 對照
& "...python.exe" dbg_boundary.py 99 95  # boundary 違反分類
```

---

## 已知 Bug / 注意事項

- **PowerShell 用分號或 `if ($?) {...}` 連接指令，不能用 `&&`**
- **Bash 工具寫 .exe 會失敗（sandbox）** → 編譯用 PowerShell
- **constructive portfolio eval ~3 分鐘**（確定性、快）；舊 SA eval 才需 13-15 分鐘
- **constructive 輸出必須 `%.17g`**（非 %.10f）→ 否則 abutment 被捨入成虛假 shapely
  fragment（M10 修正）。新增任何被 shapely 評分的幾何輸出都要遵守
- **proxy 選擇必須用 shapely vrel**（wrapper `_proxy_metrics`），不可用 C++ METRICS 的
  vrel（union-find 1e-3 tol，與 shapely 在 ~34/100 案不一致）
- **以下為舊 SA 路線（`optimizer_claude`）遺留，僅 fallback 時相關**：
  `analyze_violations.py` 不可執行（用 `viol_breakdown.py`）；`pack_cluster_anchored`
  在程式碼但不被呼叫；GNN 推論需 torch（缺則跳過，不 crash）

---

## 檔案結構

```
FloorSet/
├── optimizer_claude.cpp    ← 舊 SA placer (C++)，含 GNN-hint；僅 constructive fallback
├── optimizer_claude.py     ← 舊 SA wrapper + GNN inference；提供 _serialize_input/_parse_output
├── optimizer_claude.exe    ← 舊 SA 編譯輸出
├── constructive.cpp        ← 🏆 主程式: 建構式定框 floorplanner (C++, B 路線重寫組員架構)
│                              + M9 two-pass wire refinement + M10 %.17g 精度 / compaction
│                              + M14 hpwl_push (free single 滑動) + M15 (boundary-axis slide)
│                              + M16 same-size swap + M17 WIRE_TIEBREAK pack-order 變體
│                              + M18 WIRE_BFS (bscore 類內 BFS-connectivity greedy attachment 重排)
│                              + M19 BFS_PIN (BFS seed attachment 加 p2b pin 權重)
│                              + M20 ORDER_SWAP (refinement 前 top-K wire items pack-order
│                              pair-swap hill-climb; M21 掃出 K=16 組合) + M23 ORDER_MOVE
│                              (OS 後 top-K 拔出重插 relocation hill-climb) + BFS_NORM /
│                              CLUSTER_ORD (death-tested, default off)
│                              deterministic; env 旋鈕 (NO_COMPACT/NO_REFINE/NO_PUSH/NO_BND_PUSH/
│                              NO_SWAP/WIRE_TIEBREAK/WIRE_BFS/BFS_PIN/BFS_NORM/ORDER_SWAP/
│                              ORDER_MOVE/CLUSTER_ORD/...) + METRICS stderr
├── optimizer_constructive.py ← 🏆 PORTFOLIO wrapper: 平行 58 profile + baseline-free
│                              shapely-proxy 選擇 (_RH=1.4; 當前最佳 1.3983, ~9.5s/case;
│                              os24/os32/om16_bfs_wt 候補在註解, +0.23% 等 runtime 規則確認)
├── dbg_hpwl_push.py        ← 🆕 M14-M16 HPWL push Python 原型: 對 portfolio JSON positions 滑
│                              free single (x,y) + boundary single (free 軸) + same-size swap +
│                              cluster-rigid (停用) + violating 修復 (停用, 死路),
│                              對 evaluate_solution 真值 (orig vs pushed);
│                              ENABLE_BOUNDARY/ENABLE_CLUSTER/ENABLE_VIOLATING/ENABLE_SWAP 旗標
├── dbg_vio_stats.py        ← 🆕 M16: 對 portfolio JSON 分類全部 violating boundary block
│                              (single/cluster/preplaced × snap-to-edge free/blocked);
│                              證明 violating 修復是死路 (34 single 全 blocked)
├── rh_sweep.py             ← 🆕 OFFLINE: 建 profile×case 真值快取(並行) + 掃 _RH/proxy 參數
│                              (最快 proxy 調參器; M13 靠它找出 _RH=1.4 命中 oracle 1.4349)
├── proxy_dbg.py            ← 🆕 單案逐 profile proxy vs 真 cost (找 proxy mis-selection)
├── profile_vs_portfolio.py ← 🆕 跑任意 env profile 逐案比 portfolio JSON, 算 oracle-min 增益
│                              (通用「新 profile 值不值得加」測試器)
├── portfolio_ceiling.py    ← 🆕 OFFLINE: 跑多 profile 算 oracle 天花板 + proxy 公式
│                              搜尋 (確認 proxy≈oracle; harness vs C++ vrel 比對)
├── dbg_constructive.py     ← constructive 單 case debug (serialize + run + bbox/bv/gf)
├── analyze_constructive.py ← 🆕 per-case violation breakdown，重算 vBd/vCl/vMb + 權重排序
│                              (與官方 eval 完全吻合，~30s，乾淨 A/B 工具)
├── dbg_boundary.py         ← 🆕 分類 boundary 違反 (single/cluster/preplaced + blocked/free)
├── dbg_area.py             ← 🆕 area_gap 分解: density(bbox/ΣA) ours vs baseline + 每 dim 比
│                              (查出 area 是 dead space 1.31 vs 1.035, 非 aspect mismatch)
├── dbg_compact.py          ← 🆕 M10 compaction Python 原型: 方向 pack + csc 選擇, 對
│                              evaluate_solution 真值算 Total Score (orig vs compacted)
├── dbg_compact_cmp.py      ← 🆕 單案 Py-原型 vs C++-binary compaction 對照 (抓出 %.10f 精度 bug)
├── proxy_analysis.py       ← OFFLINE 工具: build_opt_target_pos 等，被 dbg/analyze import
│                              (proxy selector 路線已結案，但 helper 仍被分析腳本依賴)
├── floorplan_gnn.pth       ← v1 權重 (FloorplanNet, 128 hidden, unsupervised；僅舊 SA+GNN 路線用)
├── CLAUDE.md               ← 本檔案
├── gnn_training.md         ← ML 部分文件（FloorplanNet 訓練紀錄）
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

**Sanity 結果 (2026-05-27)**：

第一次 sanity (5 batches, λ_aspect=0.01) 失敗 — aspect_loss 完全 dominate
gradient（0.01 × 865 = 8.65 vs rank=0.69），cosine LR 在 5 batches 內塌到
0.00001。修正：sanity 預設 120 samples (= 30 batches)，aspect-weight 預設
0.0 in sanity，移除 noisy `probe_unsup` diagnostic。

第二次 sanity (120 samples, λ=0.0) 結果：

| Batches | avg rank_acc | rank_loss |
|---------|--------------|-----------|
| 0-4 | 0.529 | 0.69 |
| 5-9 | 0.568 | 0.68 |
| **20-24** | **0.595** | 0.66 |
| 25-29 | 0.583 | 0.66 |

**訊號存在但弱**：30 batches 內 rank_acc 從 0.53 → 0.58 (+5%)，但 noise ±0.05。
不確定 full training (500 batches) 能否突破到 0.75+。

**下一步建議：先做 oracle BL 上限實驗（~45 分鐘）**，再決定要不要花 3h 跑
full training：

| Oracle Total Score | 結論 |
|--------------------|------|
| ≤ 1.5 | ranking 是主 lever，full training 大有意義 |
| 1.5-2.5 | 邊緣有用 |
| ≥ 3.0 | BL packer 是天花板，訓練怎麼好都白工 |

### 通用 flags（所有版本）

```bash
--sanity              # 120 samples / 30 batches / aspect-weight=0 / 不存檔
--num-samples N       # 指定樣本數
--fresh               # 不 load 既有 .pth，從零訓練
--aspect-weight L     # v3 only: 預設 0.001 (sanity 0.0)
```

> ⚠️ **本環境禁止跑訓練**，若要訓練請複製到另一個 GPU 環境執行

---

## 給下一個 session 的優先建議

### 🏆 最高優先：強化 placer + portfolio + proxy（2026-06-12，當前 **1.3983**，M23 ORDER_MOVE）

`optimizer_constructive.py` portfolio 已是新主力，**反超組員所有 legit 版本、突破 < 1.40**。
M4–M23 累計大幅改善；當前 **58 profile portfolio 1.3983**（proxy 命中 oracle ceiling）。

**✅ 已完成（M4–M23）**：
- ~~MIB 統一 / cluster layout key / wire ×2000 / anchored cluster~~（單 placer 1.7045）
- ~~7→11→13 profile portfolio + baseline-free proxy~~ → 1.7045→1.6060→1.5842→1.5659
- ~~M9 two-pass wire refinement（攻 HPWL gap）~~ → 單 base 1.658、portfolio 1.5375
- ~~frame_fine profile（tighter outline）~~ → 1.5375→1.5362（marginal -0.08%）
- ~~M10a 輸出精度 %.10f→%.17g（消虛假 fragment）~~ → 單 base 1.658→**1.5532（-6.3%）**
- ~~M10b boundary 保持 compaction + csc 選擇~~ → 單 base→1.5335；**portfolio 1.5362→1.4528（-5.4%）**
- ~~M11 迭代 compaction (pack_x→pack_y→pack_x…, ICCAD_COMPACT_ITERS=8)~~ → 單 base 1.5335→1.5306；**portfolio 1.4528→1.4502（-0.18%）**
- ~~cluster-rigid compaction~~ ❌ 已試失敗（revert）
- ~~M12 40-profile 組合擴充~~ → **portfolio 1.4502→1.4371（-0.87%），< 1.43 目標達成**
- ~~M13a +4 narrow-frame profile（aspect 0.55-0.28，攻水平 dead space）~~ → 1.4371→1.4369（proxy 選不出）
- ~~M13b proxy _RH 1.0→1.4（補償 hmin/hbase）~~ → **portfolio 1.4369→1.4349，proxy = oracle ceiling**
- ~~preplaced-aligned frame（攻 case 89）~~ ❌ 已試失敗（greedy pack 不下 tighter width）
- ~~M14 post-placement HPWL push（free single → L1-median, default-on, downside-free）~~ →
  **portfolio 1.4349→1.4253（-0.67%），單 base 1.5306→1.5219，攻 hgap（最大 lever）**
- ~~M15 HPWL push 擴大可移範圍（+boundary-axis slide, default-on, downside-free）~~ →
  **portfolio 1.4253→1.4236（-0.12%），單 base 1.5219→1.5202（vBd/vCl/vMb 270/62/0 不變）**
- ~~cluster-rigid slide in HPWL push~~ ❌ 已試失敗（無 slack + FP 破壞 abutment，淨 +0.004% 1 退步）
- ~~M16 HPWL push 加 same-size swap（default-on, downside-free）~~ →
  **portfolio 1.4236→1.4231（-0.04%），單 base 1.5202→1.5197（vBd/vCl/vMb 270/62/0 不變）**
- ~~violating boundary 修復~~ ❌ M16 實測死路（0/34 可修，全 blocked；`dbg_vio_stats.py`）
- ~~profile knob 軸（WIRE_MULT 4/6、ANCHOR 0.30、ultra-narrow…）~~ ❌ M16 掃描枯竭（≤+0.063%）
- ~~per-frame compaction + csc 重估 frame~~ ❌ M17 實測死路（default-on 單 base 退步
  1.5197→1.5293：csc 跨 outline 失準拿 vCl 換 vBd；當 profile 僅 +0.008% — code 已移除）
- ~~M17 WIRE_TIEBREAK pack-order 軸 + 2 profiles（44→46）~~ →
  **portfolio 1.4231→1.4202（-0.20%）**：bscore 第一鍵不動、同類內 total_wire 大者先放；
  wtb_tall_wire **破解 case 79**（1.706→1.597）、wtb_wire 拿 **case 98**（1.4502→1.4413）
- ~~WT knob 組合掃描~~ ❌ M18 掃 7 個（anc_lo/tight/asp7/W2+anc/tight+W2/asp7+W2）全 ≤0.03%
  → WT 軸枯竭；唯一過門檻 WT+tall+anc +0.089%（case 87 獨有）已加
- ~~M18 WIRE_BFS pack-order 軸 + 3 profiles（46→49）~~ →
  **portfolio 1.4202→1.4138（-0.45%，M14 以來最大）**：bscore 類內 BFS-connectivity greedy
  attachment 重排（與已排 items+preplaced 連接最強者先放、tie 保基序、可疊 WT/frame）；
  `bfs_wt_wire` +0.316% 拿 **case 98**（1.4413→1.4221）+95/91/50、`bfs_tall_wire` +0.251%
  零重疊再破 **case 79**（1.597→1.525）+86/74/97/89、`wtb_tall_anc` +0.071%（case 87）；
  16/16 預估贏案全 realize（proxy = oracle 再驗證）
- ~~BFS knob 組合掃尾~~ ❌ M19 掃 8 個（asp5/asp7/WT+tall/narrow+anc/WT+narrow/WT+asp5/
  tight/anc）僅 tight+W2 +0.061% 過門檻；anc+W2 +0.063% 被 PIN 蓋過
- ~~M19 BFS_PIN pin-anchored seed + 2 profiles（49→51）~~ →
  **portfolio 1.4138→1.4105（-0.23%）**：BFS seed attachment 加 p2b pin 權重（pins 是
  固定 anchor，同 preplaced）；`bfs_pin_wt_wire` +0.269% 再殺 **case 95**（1.2995→1.2767）
  + **94**（1.3656→1.3411）+64、`bfs_tight_wire` +0.061% 拿 case 91 + 小案；
  13/14 realize（miss 僅 case 53 差 0.0009/權重 0）
- ~~BFS_NORM（attach/√area）+ PIN 補掃~~ ❌ M20 兩軸死路（NORM 4 組合 ≤0.026%、PIN
  tight/narrow ≤0.030%，主贏案全是 case 66 ~0.03% 單案不過門檻）
- ~~M20 ORDER_SWAP pack-order hill-climb + 1 profile（51→52）~~ →
  **portfolio 1.4105→1.4080（-0.18%）**：每 frame refinement 前對 top-8 total_wire items
  28 對做 pack-order 互換（pack-once 比較、嚴格改善才收、items_base 隔離）— greedy +
  force-directed 做不到的 jump move；`os_pin_wt_wire` +0.252% 再殺 **case 94**
  （1.3411→1.3128）+ **98**（1.4221→1.4118）+ **79**（1.5249→1.5135）+50/28/9；
  OS8 單獨僅 +0.030%（12 個低權重贏案）→ swap 需要好的起始 order；12/12 realize
- ~~M21 ORDER_SWAP 組合掃描 + 3 profiles（52→55）~~ →
  **portfolio 1.4080→1.3998（-0.58%，< 1.40 達成）**：純 portfolio 層（無 C++ 改動），
  掃 6 個 OS 組合取 3 — `os16_pin_wt_wire` +0.460%（**K=16 = 120 對 swap 池**，案例同
  PIN+WT 排序：case 98 1.4118→**1.3841**、79 1.5135→**1.4219**、82、**89 首次鬆動**
  1.8273→1.8155）、`os_bfs_wt_wire` +0.262%（無 PIN BFS 起點：86/95/97）、
  `os_bfs_tight_wire` +0.221%（**首殺硬 case 85** 1.6606→1.6255 + 42/40）；
  三者贏案近零重疊；OS12+PIN/OS16+tall/OS8+WT+tall 被蓋過不加；12/12 realize
  （掃描預估 1.3999 vs 實際 1.3998）；runtime 5.10→8.36s/case
- ~~M22 OS K 放大掃描收尾 + 2 profiles（55→57）~~ →
  **portfolio 1.3998→1.3987（-0.08%）**：`os16_bfs_wt_wire` +0.056%（磨深 86/82 + 50
  1.2920→1.2419）、`os16_bfs_tight_wire` +0.057%（**首殺 case 62** 1.6214→1.5248 +
  55/51 + 66 半收 1.4378）；8/8 realize；**K 軸飽和證明**（OS24/32 贏案中 98/79/82
  絕跡）；**runtime 風險決策**：os32_pin（71/89）+os24_pin（**66 1.3989 唯一解**）四個
  全加 live = 1.3979 但 21.9s/case → 回退（RuntimeFactor cross-submission median 未知，
  0.04% 不值 20-35% 懲罰敞口），候補在 code 註解；timeout 55→120s
- ~~M23 ORDER_MOVE relocation 軸 + CLUSTER_ORD + 1 profile（57→58）~~ →
  **portfolio 1.3987→1.3983（-0.03%）**：`ICCAD_ORDER_MOVE=K` 拔出重插 jump move（在
  ORDER_SWAP 後跑，swap 固定其他位置、move 位移中間段 — 不同的結構移動）；
  `om8_pin_wt_wire` +0.041%（**case 89 歷來最深** 1.8155→**1.8061**，勝 os32 候補
  1.8093 + 57 1.4011→1.3689 + 26/35/33）；6/6 realize；9.49s/case。
  **om16_bfs_wt_wire +0.148% 驗證有效但候補化**（59-prof live = **1.3968**、20/20
  realize：**case 96 首殺** 1.3336→1.3160 + **66 收割** 1.4378→1.3951 + 91/42/53，
  但 avg 13.51s/case；**懲罰比 = (13.51/8.8)^0.3 = +13.8% 與 median 無關** → 不值
  +0.14%）；OM16+PIN +0.040%/OM12 +0.012%（非線性丟 96/66）/OM8+OS16 +0.070%（被蓋）
  /OM8+tight 0%；**❌ CLUSTER_ORD 死路**（複合 cluster item 類內最前/最後，兩排序全
  0.000%，env-gated code 保留勿重掃）

**關鍵現況：pack-order 軸七連勝後 order 層全收割（M22 OS K 軸 + M23 OM 軸）；
knob/輕變體空間徹底掃完（M16-M20/M23 逐軸驗證 ❌）→ 增益要從 placement 層新行為來。
violation 殘留（123 cluster + 45 preplaced + 34 blocked single）後處理不可修。**
下一步：按 ROI：
1. ~~proxy _RH 修正~~ ✅ M13、~~迭代 compaction~~ ✅ M11、~~profile 擴充~~ ✅ M12、
   ~~preplaced-frame~~ ❌、~~HPWL push 三連發~~ ✅ M14/M15/M16、~~cluster-rigid slide~~ ❌、
   ~~violating 修復~~ ❌、~~per-frame csc~~ ❌、~~WIRE_TIEBREAK 軸~~ ✅ M17、
   ~~WT 組合 + WIRE_BFS~~ ✅ M18、~~BFS 組合 + BFS_PIN~~ ✅ M19、
   ~~BFS_NORM / PIN 補掃~~ ❌ M20、~~ORDER_SWAP 軸~~ ✅ M20、~~OS 組合掃~~ ✅ M21、
   ~~OS K 放大 + OS16 移植~~ ✅ M22（K 軸枯竭）、~~ORDER_MOVE 軸 + CLUSTER_ORD~~
   ✅/❌ M23（om8 ship、om16 候補、CO 死）
2. **🔑 placement 層新行為軸**：refinement pair-relocation（refinement 迴圈內拔出
   重插 slot）、compaction 方向偏好 / pack over-spread 軸先（dbg_area：寬度過寬）、
   pack 向 connectivity 重心。
3. **runtime 候補（+0.23% 現成）**：os32_pin + os24_pin（1.3979）+ om16_bfs_wt_wire
   （1.3968，M23）全部 live 驗證 → 官方 RuntimeFactor 規則確認寬鬆即加回。
   **懲罰比公式（M23）：兩配置 factor 比 = (t1/t2)^0.3，與未知 median 無關**。
4. **case 66 半收**（1.4378，om16 可到 1.3951）；**case 89**（1.8061 M23 新低）；
   **case 96**（1.3336，om16 可到 1.3160）；85（1.6255）。⚠️ frame 偏好「不超出
   preplaced 外緣」已試失敗（M13）。
5. **次要**：掃 `ICCAD_REFINE_ITERS`、`ICCAD_PUSH_PASSES`。⚠️ 已驗證**不是 lever**：
   `layout_score` hpwl 權重、frame scale 細化。
6. ⚠️ runtime ~9.5s/case（58 profile）。**本地 eval 強制 RuntimeFactor=1.0 中性**；官方
   算 cross-submission median（組員 portfolio ~11s 唯一參考）→ **8-11s 是安全帶，>13s
   依懲罰比公式必虧**（M22/M23 決策先例）。單 placer 改進所有 profile 同步受惠。

> ⚠️ **試過會退步**：max_trials 試「所有 frame」→ 2.42；BP_WEIGHT 拉高無效；
>    wire ×50000 反彈 1.93；proxy near-tie min-vrel tiebreak 反而更差（proxy 夠準）。
> ⚠️ **proxy 必須用 shapely vrel**（wrapper `_proxy_metrics`），不能用 C++ METRICS
>    的 vrel（union-find 1e-3 tol，與 shapely 差 34/100 案 → 退到 1.6388）。
> ✅ 工具：`portfolio_ceiling.py`（oracle 天花板 + proxy 搜尋，~5min）；
>    `analyze_constructive.py`（單 profile per-case breakdown，~30s）；
>    `dbg_boundary.py <ids>`；`dbg_vio_stats.py`（portfolio 違反分類統計）；`dbg_constructive.py`；
>    `dbg_hpwl_push.py`（M14-M16 push 原型, ENABLE_BOUNDARY/ENABLE_CLUSTER/ENABLE_VIOLATING/
>    ENABLE_SWAP 旗標）；`profile_vs_portfolio.py`（新 profile 候選離線掃描）。
>    env 旋鈕：`ICCAD_WIRE_MULT` / `ICCAD_ANCHOR_W` / `ICCAD_LR_ASPECT` / `ICCAD_TB_ASPECT` /
>    `ICCAD_BP_WEIGHT` / `ICCAD_NO_COMPACT=1`（關 M10 compaction）/ `ICCAD_NO_REFINE=1` /
>    `ICCAD_NO_PUSH=1`（關 M14-M16 HPWL push）/ `ICCAD_NO_BND_PUSH=1`（關 M15 boundary-axis,
>    退回 M14 free-single only）/ `ICCAD_NO_SWAP=1`（關 M16 swap, 退回 M15）/ `ICCAD_PUSH_PASSES=N` /
>    `ICCAD_WIRE_TIEBREAK=1`（M17 pack-order: bscore 同類內 total_wire 大者先放）/
>    `ICCAD_WIRE_BFS=1`（M18 pack-order: bscore 類內 BFS-connectivity greedy attachment
>    重排，可疊 WT/frame knob）/ `ICCAD_BFS_PIN=1`（M19: BFS seed attachment 加 p2b
>    pin 權重，需配 WIRE_BFS）/ `ICCAD_BFS_NORM=1`（M20 死路，留 code 勿重掃）/
>    `ICCAD_ORDER_SWAP=K`（M20: refinement 前 top-K wire items pack-order pair-swap
>    hill-climb，獨立軸可疊任何排序）/ `ICCAD_ORDER_MOVE=K`（M23: OS 後 top-K items
>    拔出重插 relocation hill-climb，獨立軸）/ `ICCAD_CLUSTER_ORD=1/2`（M23 死路，
>    留 code 勿重掃）；
>    `ICCAD_CONSTRUCTIVE_SINGLE=1` 退回單 base。analyze/dbg 直跑 exe（不經 wrapper）→
>    量單一 profile。組員參考碼在 `C:\Users\Nordra\Downloads\teammate_iccad_study\`。

### （舊）給下一個 session 的優先建議

> ⚠️ **2026-05-26 更新**：以下舊建議（boundary / bbox / slack）都是 optimization
> 思維。範式已轉移到 reconstruction，請先看本文件頂部「🚨 範式轉移」段落，
> 再決定要不要繼續走 optimization 路線。

### 新優先級（2026-05-31 oracle 實驗後 — 結論：ML ranking 死路）

#### Oracle BL 上限實驗結果 (oracle-perm 腳本, code 已刪, 2026-05-31)

| Mode | Total Score | Avg Cost | 解讀 |
|------|------------|----------|------|
| 現狀 baseline (no GNN) | 3.4308 | 2.6478 | 參照線 |
| 現狀 + v1 GNN | 3.3258 | 2.6548 | 我們最佳 |
| **Oracle perm + SA (exe)** | **3.2673** | **2.6494** | **vs 現狀只進步 1.8%** |
| Oracle perm + oracle shape, no SA (bl) | 9.9996 | 9.8569 | BL packer 完全失敗 |
| fp_sol verbatim (raw) | 1.1079 | 1.1097 | 對上 teammate 的 1.1079 |

→ 判讀表 **3.27 ≥ 3.0 → BL packer 是天花板，v3 ranking ML 訓練是白工**
→ 即使 ML 給出完美 perm，pipeline 上限是 3.27 (距 1.6 legit / 1.0 oracle 都很遠)
→ Raw vs exe 的 1.10 → 3.27 巨幅落差 = 我們的 placer + shape 邏輯崩潰，
   ranking 救不了

#### 結論：路線徹底轉向 placer 改造

1. ❌ **v3 ML ranking 訓練：放棄**（上限 1.8%，不值得燒 GPU）
2. ❌ **Port 組員 v5 boundary aspect**（已試, 2026-05-31, **失敗**）
   - LEFT/RIGHT-only blocks: aspect 2.50；TOP/BOTTOM-only: 0.40
   - 結果：Total Score 3.3258 → **3.4255 退步 3%**
   - 退步集中在 n≥80 cases (n=80-99 avg 3.03、n=100-119 avg 3.32)
   - 根因：我們 skyline BL packing 不像 teammate 的 shelf packing 能利用
     tall edge blocks；前者讓 tall block 在左邊形成 cliff 害後續 block 找位
   - 已 revert，僅在 optimizer_claude.cpp 留註解避免重試
3. ✅ **Portfolio selector**（已實作, 2026-05-31, **成功**；code 後已刪）
   - Portfolio wrapper：4 profile 並行（ThreadPool + 4 subprocess）
   - 每 profile full 8s SA，wall time 8.6s（16 cores）
   - Contest-shape proxy 挑最佳：`(1+α(area_gap+hpwl_rel))·exp(β·v_rel)`
   - 結果：3.3258 → **3.1584 (-5.0%)**, Avg Cost 2.6548 → 2.4759 (-6.7%)
   - 100/100 feasible
4. ✅ **Portfolio 擴充 v1**（已實作, 2026-06-01, **成功**）
   - 加 `pin_centroid` / `degree_desc` / `degree_asc` 共 7 profile
   - 3.1584 → **3.1082 (-3.2%)**，100/100 feasible
   - 新 profile 拿下 52/100 cases (degree_desc 21, pin_centroid 16, degree_asc 15)
5. ✅ **Portfolio 擴充 v2 + W_BOUNDARY=100**（已實作, 2026-06-01, **成功**）
   - 加 `high_boundary` (connectivity perm + W_BOUNDARY=100 via env var)
   - C++ `main()` 新增 `ICCAD_W_BOUNDARY` env var override 支援
   - 3.1082 → **3.0625 (-1.5%)**，100/100 feasible
   - `high_boundary` 拿下 18/100 (overall #1)，主導大 n cases
     (n=100-109 4/10, n=110-119 3/10, n=120 1/1)
6. ❌ **Portfolio 擴充 v3 + W_VIOL 變體**（已試, 2026-06-02, **失敗**）
   - C++ `main()` 新增 `ICCAD_W_VIOL_MULT` env var（校正後乘上 W_VIOL）
   - 加 `low_viol` (×0.5) / `high_viol` (×2.0) → 10 profile
   - 同機 clean A/B：8-prof **3.0554** vs 10-prof **3.0859**（**退步 +1.0%**）
   - 新 profile 拿下 19/100 case (low_viol 9 + high_viol 10)、佔 13.9% 加權，
     **但 proxy 在 <1.5% margin 的 near-tie 選錯** → 淨退步
   - **結論：profile 數飽和在 8，瓶頸轉移到 proxy selector 準度**
   - 已 revert default 回 8；env-var 基建 + low_viol/high_viol 定義保留，
     可經 `ICCAD_PORTFOLIO_PROFILES` 在改良 proxy 後重測
7. ✅ **proxy near-tie tie-break**（已試, 2026-06-02, **小贏並 ship**）
   - 建 `proxy_analysis.py`（仍在）+ 掃公式腳本（已刪）量測 oracle 天花板
   - **Oracle-selector 天花板 = ~3.03**（同 run proxy 3.1288 vs oracle 3.0335,
     gap 3.0%）→ **完美選擇也破不了 3.00，selection 路線已盡**
   - 有效 lever 僅 near-tie min-viol tie-break (margin 0.02)：離線固定輸出
     A/B 3.1288 → **3.0979 (-1.0%)**。已實作 `_pick_best`，
     env `ICCAD_PROXY_TIE_MARGIN`。area_c/hpwl_c/α/β 皆非 lever
   - ⚠️ live single-run 量不到（±2-3% 限時噪音 > 1% 改動，見目標 2 根因）
8. ❌ **shape prediction ML**（已試, 2026-05-31, **失敗**）
   - oracle 實驗：oracle shape only 3.42 (改善 0.3%)、shape+perm 3.37
   - 結論：即使給完美 shape，pipeline 上限就是 ~3.4
   - Shape 不是 lever，跟 perm ML 一樣被 placer 架構 cap 住
9. ❌ **oracle 路線**：1.0322 在 hidden test 不存在

### 失敗紀錄（避免重蹈）

- **v2 (2026-05-26) supervised MSE on fp_sol absolute position**：
  ill-posed 一對多任務，loss 震盪、預測佈局物理無效（unsup_cost 暴衝到 47M）
  → `floorplan_gnn_v2.pth` 已捨棄，**不要 commit**

### 舊優先級（optimization 範式，已 deprecate）

~~第一件事：W_BOUNDARY 100 → 10~~（已做）
~~第二件事：viol_breakdown 確認 vBd 主導~~
~~第三件事：slack=0 boundary block 處理~~
~~第四件事：bbox shrinking~~

這些只能在「我們要繼續做 optimization」的前提下有意義。在 reconstruction
範式下，所有 violation/HPWL/area 都是 baseline 內建的副產品 — 還原原圖
自然就沒問題。
