# ICCAD 2026 FloorSet — Session Context

## 🚨 範式轉移 (2026-05-26, 重要！)

**這個競賽不是 floorplan optimization，而是 reconstruction（拼圖還原）。**

### 證據（含 2026-05-27 修正）

組員的 1.0322 經查證 **是 oracle**（讀本地 validation label），hidden test 退回 fallback。
真正的 legit 上限約 **1.6**（無 label 的 portfolio 方法）。

- `teammate_eva/v8_puzzle_fingerprint_oracle.py:68` — `FloorplanDatasetLiteTest("../")`
  → 讀本地 label，fingerprint 比對輸入；命中回傳 ground truth，沒命中退回 `my_optimizer.py`
- 競賽 server hidden test 沒 label → 100% fallback，分數就是 `my_optimizer.py` 的分數

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

### 🏆 最佳已驗證版本：Total Score = **1.6060** (Constructive portfolio M6, 2026-06-05)

**反超組員所有 legit 版本（含 v6/v7 portfolio ~1.62）。** `constructive.cpp` +
`optimizer_constructive.py` 是組員 `my_optimizer.py` 建構式定框 floorplanner 的
C++ 重寫（B 路線）+ 我們自建的 portfolio 選擇層。100/100 feasible，~0.43s/case。
確定性（無 randomness/限時 → run-to-run 一致，可精確 A/B；官方 eval 與離線預測
完全吻合 1.6060 = 1.6060）。

**對標**：組員 v5 = 1.7429（好 7.9%）、組員 v6/v7 portfolio ~1.62（**反超**）。

**M6 portfolio 層**（`optimizer_constructive.py`）：平行跑 **7 個 deterministic
profile**（env 旋鈕變體：base / wire_hi / anc_lo / area_lean / aspect_hi(3.5) /
aspect_xhi(5.0) / asp_wire），用 **baseline-free proxy** 選最佳。
- proxy = `(area/Â + hpwl/hmin)·exp(2·vrel)`，Â=1.035·ΣblockArea，hmin=該 case 各
  profile 最小 hpwl（推導自 cost=0.5(area/A+hpwl/H)·exp(2·vrel)，vrel 精確、area
  baseline 可估、hpwl baseline 用 per-case 尺度）
- ⚠️ **vrel 必須用 shapely 算**（比照 harness）：C++ `count_group_fragments`
  用 1e-3 tol 把 MARGIN 間隙當接觸，與 shapely 在 **34/100 案不一致** → 若用 C++
  vrel 選擇，退到 1.6388。wrapper 內 `_proxy_metrics` 用 shapely 重算 → 命中 1.6060
- **oracle 天花板 = 1.6057，proxy 抓到 1.6060（幾乎滿分）**：deterministic 無 SA
  限時噪音是關鍵（舊 SA portfolio proxy 只抓到天花板的一半）
- aspect 變體是最強分歧軸（高 LR aspect 降 vBd、升 hpwl → 在 violation-heavy case 贏）
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
+wire ×2000 1.8218 → M5: +anchored cluster 1.7045 → **M6: +7-profile portfolio
1.6060**（本 session 共 **-28.7%**）。

**下一步（→ 壓低 oracle 天花板 1.6057）**：portfolio proxy 已幾乎滿分（抓到天花板
99.9%），再進步要**降低天花板本身** = 更好/更分歧的 profile，或更強的單一 placer。
單一 base 殘留 vBd（少數 preplaced boundary block 位置固定無解）+ area_gap ~0.28。
詳見「給下一個 session」。

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
| **2026-06-05 constructive M6 (+7-profile portfolio + baseline-free proxy)** | **1.6060** ← **新最佳, -5.8%, 反超組員 v6/v7 (~1.62); oracle 天花板 1.6057** |
| 【外部驗證】組員 my_optimizer.py 餵我們 evaluator | 1.7429 ← 確認架構可移植 |
| 2026-05-31 oracle shape only (sanity)  | 3.4199 ← **shape ML 死** (改善 0.3%) |
| 2026-05-31 oracle shape + oracle perm | 3.3672 ← 鎖死 shape 反害 SA |
| 2026-05-26 v2 supervised MSE on fp_sol (2000 sample, < 3h) | **失敗** — pos_mse 震盪、unsup_cost 47M，.pth 已棄 |
| 2026-05-27 v3 sanity (120 sample, 30 batches) | rank_acc 0.53 → 0.58，訊號弱 — 待 oracle 實驗決定 |
| **2026-05-31 oracle perm + SA (上限實驗)** | **3.2673** ← BL packer 是天花板，v3 ML 放棄 |
| **【外部參考】組員 v6/v7 portfolio (legit, 無 label)** | **~1.62** ← 真正可達目標 |
| 【外部參考】組員 v9 oracle (讀 label) | 1.0322 ← hidden test 不適用 |

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
- ~~`W_BOUNDARY = 100`~~ ✅ 已完成，已成為 `high_boundary` profile
- 還有 4 cores 空，可加更多 portfolio profile

---

## 預期目標

> ⚠️ 範式轉移後，optimization 思維的舊目標（boundary count、slack 處理）
> 重要性降低。當前路線是 portfolio scaling + placer 架構演進。

### 已達成
- ~~目標 1: boundary violations 中位數 ≤ 10~~（仍未達，但靠 high_boundary 局部改善）
- ✅ 目標 2: Total Score < 3.20（**3.0625 達成**）
- ✅ 目標 3: W_BOUNDARY = 100 實驗（**成功，已整合**）
- ✅ 目標 6: Total Score < 3.10（**3.0625 達成**）

### 短期（1–3 個迭代）
- **目標 1**：Total Score < 3.00（從 3.0625 再降 2%）
  - ❌ ~~portfolio 擴充 v3（W_VIOL 變體）~~ **已試 2026-06-02, 失敗**：
    加 low_viol(×0.5)/high_viol(×2.0) → 10 profile，同機 clean A/B
    8-prof 3.0554 vs 10-prof 3.0859（**退步 +1.0%**）。新 profile 雖拿下
    19/100 case，但 contest-shape proxy 在 <1.5% margin 的 case 選錯
    （挑了 true-cost 較差的）。**profile 數已飽和在 8。**
  - ✅ **proxy selector 改良（已試 2026-06-02, 小贏並 ship）**
    - 建工具：`proxy_analysis.py`（跑全 8 profile，用 harness 自己的
      `evaluate_solution` 算每個 profile 的**真實 cost**，dump 出真實-cost JSON）
      + 離線掃 proxy 公式評分腳本（後兩者 code/dump 已刪，僅留結論）
    - 🔑 **Oracle-selector 天花板 = ~3.03**（8-profile 集合, 同 run proxy 3.1288
      vs oracle 3.0335, gap 3.0%）→ **完美選擇也破不了 3.00**。proxy tuning
      ROI 上限就是這 3.0%，且實際抓不到全部
    - 診斷：area baseline 1.035 已正確（A/ΣA median 1.03）；HPWL baseline
      無穩定常數（H/best_h range [0.43,0.94]）→ hpwl_c / area_c 都不是 lever
    - **唯一有效 lever：near-tie → min-violation tie-break**（margin 0.02）：
      離線固定輸出 A/B 3.1288 → **3.0979 (-1.0%)**。原理：proxy 的 HPWL/area
      baseline 是估計值，sub-% margin 不可信，但 v_rel 是精確算出的 → near-tie
      時 defer 給 v_rel。已實作於 `_pick_best`，env `ICCAD_PROXY_TIE_MARGIN`
      (default 0.02)，α/β 保持真實 cost 值 (0.5/2.0) 避免 overfit
    - ⚠️ **不可用 target_positions/fp_sol 當 baseline**（= oracle，hidden test 沒）；
      `solve()` 只收 preplaced 位置 + fixed shape，free block GT 不給（無 leak）
  - 🔑 **結論：селection 已盡，破 3.00 要靠 placer 架構 (目標 5) 或更好 profile**
  - 次路徑：repair-style 後處理（見目標 4）、placer 架構（見目標 5）
- **目標 2**：驗證 ±2% 變異 — **已查明根因**
  - 同機觀察 8-prof: 3.0554 / 3.1288 / 3.1040（同 code 跨 3 run，spread 2.4%）
  - **根因：C++ rng seed 固定 (42) 但 SA 是 wall-clock 限時 (8s) 非 iteration 限**
    → 平行 8-10 subprocess 競爭 CPU，每 run 8s 內跑的 iteration 數不同 → 結果飄
  - 含意：**單一 live run 無法量測 < 2-3% 的改動**，要用離線固定輸出 A/B
    （fixed-output 模式）。或改 SA 為 iteration-limited 讓它 deterministic
    （未做，屬 placer 改動）

### 中期（4–6 個迭代）
- **目標 3**：Total Score < 2.80（需突破 placer 架構）
- **目標 4**：實作 repair-style 後處理（針對性修 violation，
  類似組員 v5 後的 boundary nudge / single-edge escape）
- **目標 5**：探索 shelf packing 或 B*-tree 取代當前 skyline BL packer
  （組員 v5/v6 用 shelf 達 1.76/1.62）

### 長期
- **目標 6**：縮小到 legit teammate 1.6 範圍（當前 3.06 / 1.6 = 1.9× gap）
- **目標 7**：把 SA 角色從「找解」改成「微調 ML 輸出」

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
├── constructive.cpp        ← 🏆 建構式定框 floorplanner (C++, B 路線重寫組員架構)
│                              單 profile ~0.16s/case, deterministic; env 旋鈕 +
│                              METRICS stderr 輸出 (現由 wrapper 用 shapely 重算取代)
├── optimizer_constructive.py ← 🏆 PORTFOLIO wrapper: 平行 7 profile + baseline-free
│                              proxy 選擇 (當前最佳 1.6060, ~0.43s/case)
├── portfolio_ceiling.py    ← 🆕 OFFLINE: 跑多 profile 算 oracle 天花板 + proxy 公式
│                              搜尋 (確認 proxy≈oracle; harness vs C++ vrel 比對)
├── dbg_constructive.py     ← constructive 單 case debug (serialize + run + bbox/bv/gf)
├── analyze_constructive.py ← 🆕 per-case violation breakdown，重算 vBd/vCl/vMb + 權重排序
│                              (與官方 eval 完全吻合，~30s，乾淨 A/B 工具)
├── dbg_boundary.py         ← 🆕 分類 boundary 違反 (single/cluster/preplaced + blocked/free)
├── proxy_analysis.py       ← OFFLINE 工具: build_opt_target_pos 等，被 dbg/analyze import
│                              (proxy selector 路線已結案，但 helper 仍被分析腳本依賴)
├── floorplan_gnn.pth       ← v1 權重 (FloorplanNet, 128 hidden, unsupervised；僅舊 SA+GNN 路線用)
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

### 🏆 最高優先：壓低 oracle 天花板（2026-06-05，當前 **1.6060**，天花板 1.6057）

`optimizer_constructive.py` portfolio 已是新主力，**反超組員所有 legit 版本**。本
session 完成 M4–M6 共 -28.7%（見上方狀態段）。

**✅ 已完成（M4–M6）**：
- ~~MIB 統一~~ → vMb 145→0，-3.7%
- ~~cluster layout key (frag/boundary 排 area 前)~~ → vBd 528→357，-9.4%
- ~~wire weight ×2000~~ → hpwl 大降，-7.2%
- ~~anchored cluster first-pass~~ → vBd 359→307 vCl 260→208，-6.4%
- ~~7-profile portfolio + baseline-free proxy~~ → 1.7045→1.6060，-5.8%

**關鍵現況：proxy 已抓到天花板 99.9%（1.6060 vs oracle 1.6057）。** 純擴 profile /
調 proxy 已無空間 → **要進步必須降低天花板本身**，按 ROI：
1. **更分歧的 profile（降天花板，最高 ROI）**：7 profile 的 oracle=1.6057。加結構性
   差異更大的 profile 會降天花板（aspect 是已知最強軸）。可加：frame aspect/scale 變體
   （需把 `frame_candidates` 的 aspects/scales 開成 env 旋鈕）、cluster ordering 變體、
   不同 boundary penalty 結構。**流程**：在 `portfolio_ceiling.py` 加 profile → 看
   oracle 降多少 → 降得動才加進 wrapper `_PROFILES`（proxy 幾乎滿分，天花板降多少幾乎全拿）
2. **強化單一 placer（同時降所有 profile + 天花板）**：
   - **vBd**：少數 preplaced boundary block 位置固定無解 → frame 選擇偏好不超出其外緣
   - **area_gap ~0.28**：補 grid/top_aligned/right_aligned cluster layout；小 n frame scale 加細
3. ⚠️ runtime 0.43s/case（7 profile 平行 + shapely vrel）。加很多 profile 要注意核心數

> ⚠️ **試過會退步**：max_trials 試「所有 frame」→ 2.42；BP_WEIGHT 拉高無效；
>    wire ×50000 反彈 1.93；proxy near-tie min-vrel tiebreak 反而更差（proxy 夠準）。
> ⚠️ **proxy 必須用 shapely vrel**（wrapper `_proxy_metrics`），不能用 C++ METRICS
>    的 vrel（union-find 1e-3 tol，與 shapely 差 34/100 案 → 退到 1.6388）。
> ✅ 工具：`portfolio_ceiling.py`（oracle 天花板 + proxy 搜尋，~5min）；
>    `analyze_constructive.py`（單 profile per-case breakdown，~30s）；
>    `dbg_boundary.py <ids>`；`dbg_constructive.py`。env 旋鈕：`ICCAD_WIRE_MULT` /
>    `ICCAD_ANCHOR_W` / `ICCAD_LR_ASPECT` / `ICCAD_TB_ASPECT` / `ICCAD_BP_WEIGHT`；
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
10. **不要動 `teammate_eva/` 內任何檔案** — 由使用者管理

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
