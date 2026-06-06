# ICCAD 2026 FloorSet — Session Context

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

### 🏆 最佳已驗證版本：Total Score = **1.4528** (Constructive portfolio M10: 精度修正 + compaction, 2026-06-06)

**反超組員所有 legit 版本（含 v6/v7 portfolio ~1.62，現 -10%）。** `constructive.cpp` +
`optimizer_constructive.py` 是組員 `my_optimizer.py` 建構式定框 floorplanner 的
C++ 重寫（B 路線）+ 我們自建的 portfolio 選擇層。100/100 feasible，~1.63s/case（14
profile）。確定性（無 randomness/限時 → run-to-run 一致，可精確 A/B；官方 eval 1.4528）。

**M10（本 session，2026-06-06，攻 area_gap dead space）= 兩件事，皆驗證有效：**

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
1.5659 → M9: +two-pass wire refinement 1.5375 → **M10: 精度修正 + compaction 1.4528**
（M10 portfolio **-5.4%**；M4 起累計 **-34%**）。

**下一步（→ 繼續壓低天花板，當前 1.4528）**：M10 修掉精度漏分 + 初步 compaction。
compaction 只跑 4 個 axis-aligned 方向 pack（單次），**area dead space 只部分回收**
（density 仍 >1.1）。續壓 area 的試法（按 ROI）：
1. **迭代 compaction**：pack_x→pack_y→pack_x… 多輪（Y-pack 後 X 可能再開 slack）。csc
   下檔保護，安全。可能再擠 1-2%。**最低風險、先試這個。**
2. **cluster-rigid pack**：把整個 cluster 當剛體一起滑（而非逐 block），可更激進 pack 而
   不 fragment（現在逐 block pack 靠 csc 拒絕 fragment 的候選，較保守）。
3. **更多 pack 起點/順序**：先 pack over-spread 軸（dbg_area 顯示多數 case 寬度過寬
   w/wb 1.3-1.7、高度準），或 pack 向 connectivity 重心。
4. **agap outlier 個案**（79 agap 0.706、99 退大 frame）：tighter frame pack 不下，
   compaction 後可重估這些 case 是否該選更小 frame。
次大殘留：硬 case（89 hgap 0.751 + vBd 7、85 vBd 10）多為 preplaced boundary 撐壞 outline。

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
| **2026-06-06 constructive M10b (+boundary 保持 compaction, csc 選擇)** | **1.4528** ← **新最佳, portfolio -5.4%; 單 base 1.5532→1.5335; 100/100 feasible** |
| 【外部驗證】組員 my_optimizer.py 餵我們 evaluator | 1.7429 ← 確認架構可移植 |
| 2026-05-31 oracle shape only (sanity)  | 3.4199 ← **shape ML 死** (改善 0.3%) |
| 2026-05-31 oracle shape + oracle perm | 3.3672 ← 鎖死 shape 反害 SA |
| 2026-05-26 v2 supervised MSE on fp_sol (2000 sample, < 3h) | **失敗** — pos_mse 震盪、unsup_cost 47M，.pth 已棄 |
| 2026-05-27 v3 sanity (120 sample, 30 batches) | rank_acc 0.53 → 0.58，訊號弱 — 待 oracle 實驗決定 |
| **2026-05-31 oracle perm + SA (上限實驗)** | **3.2673** ← BL packer 是天花板，v3 ML 放棄 |
| **【外部參考】組員 v6/v7 portfolio (legit, 無 label)** | **~1.62** ← 真正可達目標 |
| 【外部參考】組員 v9 oracle (讀 label) | 1.0322 ← hidden test 不適用 |

---

## 這個階段想解決的問題（constructive M10 後，當前 1.4528）

> 舊 SA 範式的瓶頸（slack=0 boundary、SA 收斂、bbox shrinking）已隨架構換成
> constructive placer 而作廢。以下是**當前** placer 的瓶頸，依 leverage 排序。

### A. area_gap dead space 仍是最大 uniform 缺口（最高 leverage）
- M10 compaction（單次 4 方向 pack）後 density 仍 >1.1（原圖 1.035）→ 還有 void 可擠
- compaction 逐 block pack，靠 csc 拒絕 fragment 的候選 → 較保守，未榨乾
- 試法：迭代 compaction（pack_x→pack_y→pack_x…）、cluster-rigid pack、更聰明的
  pack 起點（先壓 over-spread 軸；多數 case w/wb 1.3-1.7 但 h/hb≈1，見 `dbg_area.py`）

### B. agap outlier 個案
- case 79 (agap 0.706)、99 等：tighter frame pack 不下 → 退到大/寬 frame
- compaction 後可**重估** frame 選擇（大 frame 易 pack + compaction 擠掉 void
  → 可能勝過原本選的 tight frame）

### C. 硬 case：preplaced boundary block 撐壞 outline
- case 89 (hgap 0.751 + vBd 7)、85 (vBd 10)：preplaced 位置固定，bbox 邊到不了它
- 結構約束 > wire/compaction 拉力 → frame 選擇宜偏好「不超出 preplaced 外緣」

### D. proxy selector / profile 多樣性
- 14-profile baseline-free proxy 已近 oracle 天花板（M8 時 proxy=oracle=1.5659）
- M10 後 proxy 需用 **shapely vrel**（wrapper `_proxy_metrics`），%.17g 後 shapely 與
  C++ 內部更一致 → 選擇更準。加 profile 仍下檔保護（無用只花 runtime）

---

## 預期目標

> 基準線：constructive portfolio **1.4528**（M10）。對標：組員 legit portfolio
> ~1.62（**已反超 ~10%**）、組員 oracle 1.0322（讀 label，hidden test 不適用）、
> fp_sol verbatim 1.1079（理論重建上限）。確定性 → 可精確 A/B，無 SA 限時噪音。

### 已達成
- ✅ Total Score < 3.00 / < 2.00 / < 1.60（constructive 路線一路下殺，當前 1.4528）
- ✅ **反超組員所有 legit 版本**（v5 1.7429、v6/v7 portfolio ~1.62）
- ✅ baseline-free proxy ≈ oracle 天花板（無 label leak，hidden test 可用）
- ✅ M10 精度修正（消虛假 fragment）+ boundary 保持 compaction（攻 area_gap）

### 短期（1–3 個迭代，續攻 area_gap）
- **目標 1**：Total Score < 1.43 — **迭代 compaction**（pack_x→pack_y→pack_x… 多輪，
  csc 下檔保護，最低風險）。density 仍 >1.1 → 有空間。
- **目標 2**：cluster-rigid pack（整 cluster 當剛體滑，比逐 block 更激進不 fragment）
  + 更聰明 pack 起點（先壓 over-spread 軸 / connectivity 重心）
- **目標 3**：agap outlier（79/99）compaction 後重估 frame 選擇 → 針對性回收

### 中期（4–6 個迭代）
- **目標 4**：Total Score < 1.35 — 需 placer 結構升級（compaction 進化到極限後）
- **目標 5**：硬 case（preplaced boundary 撐壞 outline）的 frame 偏好策略
- **目標 6**：profile 軸擴充（cluster ordering 變體），proxy 已近 oracle → 下檔保護

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

### 1. compaction 進化（攻 area_gap，當前最高 ROI）
- **迭代**：`compact_layout` 現只跑單輪 4 方向 pack。pack_y 後 X 軸可能再開 slack →
  迴圈 pack_x→pack_y→pack_x… 到收斂。csc 下檔保護，安全。先試這個。
- **cluster-rigid**：逐 block pack 會 fragment cluster（靠 csc 拒絕，較保守）。改成把
  整個 cluster 當剛體一起滑，可更激進 pack 而不破 grouping。
- **起點/順序**：先 pack over-spread 軸（`dbg_area.py` 顯示多數 case 寬度過寬）；
  或 pack 向 connectivity 重心以同時壓 hpwl。

### 2. frame 選擇與 compaction 協同
- compaction 後重估 frame：大 frame 易 pack + compaction 擠 void，可能勝過原本選的
  tight frame（特別是 agap outlier 79/99）。可在 solve() frame loop 內對每 frame
  compaction 後再比 csc（注意 overfit 風險，見「per-frame 退步」教訓）。

### 3. 硬 case 處理（preplaced boundary 撐壞 outline）
- case 89/85：frame 偏好「不超出 preplaced 外緣」，或對這類 case 特化 outline。
- 用 `dbg_boundary.py` 分類違反（single/cluster/preplaced × blocked/free）。

### 4. profile 軸擴充（下檔保護，低風險）
- 新分歧軸：cluster ordering 變體、compaction 方向偏好。proxy 已近 oracle →
  無用 profile 只是不被選、不傷分（只花 runtime）。

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
│                              deterministic; env 旋鈕 (NO_COMPACT/NO_REFINE/...) + METRICS stderr
├── optimizer_constructive.py ← 🏆 PORTFOLIO wrapper: 平行 14 profile + baseline-free
│                              shapely-proxy 選擇 (當前最佳 1.4528, ~1.63s/case)
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

### 🏆 最高優先：強化單一 placer（2026-06-06，當前 **1.5375**，天花板 ~1.5375）

`optimizer_constructive.py` portfolio 已是新主力，**反超組員所有 legit 版本**。M4–M8
共 -30.5%；M9（本 session）two-pass wire refinement 再 **-1.8%**（見上方狀態段）。

**✅ 已完成（M4–M10）**：
- ~~MIB 統一 / cluster layout key / wire ×2000 / anchored cluster~~（單 placer 1.7045）
- ~~7→11→13 profile portfolio + baseline-free proxy~~ → 1.7045→1.6060→1.5842→1.5659
- ~~M9 two-pass wire refinement（攻 HPWL gap）~~ → 單 base 1.658、portfolio 1.5375
- ~~frame_fine profile（tighter outline）~~ → 1.5375→1.5362（marginal -0.08%）
- ~~M10a 輸出精度 %.10f→%.17g（消虛假 fragment）~~ → 單 base 1.658→**1.5532（-6.3%）**
- ~~M10b boundary 保持 compaction + csc 選擇~~ → 單 base→1.5335；**portfolio 1.5362→1.4528（-5.4%）**

**關鍵現況：M10 修掉精度漏分 + 初步 compaction（單次 4 方向 pack）。area dead space
只部分回收（density 仍 >1.1），仍是最大 uniform 缺口。** 下一步續**強化單一 placer**，按 ROI：
0. **迭代 compaction（新 #1，最低風險）**：pack_x→pack_y→pack_x… 多輪，csc 下檔保護。先試。
1. ~~boundary-接觸保持的 compaction~~ ✅ M10b 已做（單次方向 pack）。可升級為 cluster-rigid
   pack（整 cluster 當剛體滑，比逐 block 更激進不 fragment）或迭代（見 0）。
2. **agap outlier 個案**（case 79 agap 0.706、99 0.392 等退到 s≥1.35 大 frame）：找為何
   tighter frame pack 不下（多半某大 block/cluster 卡住）→ 針對性處理，比 uniform 壓更省。
3. **vBd 硬 case**（89 hgap 0.751+vBd 7、85 vBd 10）：多為 **preplaced boundary block
   撐壞 outline**（位置固定，bbox 邊到不了它）→ frame 選擇偏好「不超出 preplaced 外緣」。
   refinement 幫助有限（結構約束 > wire 拉力）。用 `dbg_boundary.py` 確認佔比。
4. **次要**：剩餘 profile 軸（cluster ordering 變體）；掃 `ICCAD_REFINE_ITERS`（12-24 平緩，
   >12 邊際 <0.2%）。⚠️ 已驗證**不是 lever**：`layout_score` hpwl 權重（HW_MULT 3/8 不動
   選擇）、frame scale 細化（frame_fine 僅 -0.08%）。
5. ⚠️ runtime 1.46s/case（14 profile，M9 約 2× 因每 frame 多跑 12 refine pass）。**本地 eval
   強制 RuntimeFactor=1.0 → runtime 對分數中性**；官方 leaderboard 算 cross-submission
   median，1.46s 對 floorplanner 仍快。單 placer 改進所有 profile 同步受惠。

> ⚠️ **試過會退步**：max_trials 試「所有 frame」→ 2.42；BP_WEIGHT 拉高無效；
>    wire ×50000 反彈 1.93；proxy near-tie min-vrel tiebreak 反而更差（proxy 夠準）。
> ⚠️ **proxy 必須用 shapely vrel**（wrapper `_proxy_metrics`），不能用 C++ METRICS
>    的 vrel（union-find 1e-3 tol，與 shapely 差 34/100 案 → 退到 1.6388）。
> ✅ 工具：`portfolio_ceiling.py`（oracle 天花板 + proxy 搜尋，~5min）；
>    `analyze_constructive.py`（單 profile per-case breakdown，~30s）；
>    `dbg_boundary.py <ids>`；`dbg_constructive.py`。env 旋鈕：`ICCAD_WIRE_MULT` /
>    `ICCAD_ANCHOR_W` / `ICCAD_LR_ASPECT` / `ICCAD_TB_ASPECT` / `ICCAD_BP_WEIGHT` /
>    `ICCAD_NO_COMPACT=1`（關 M10 compaction）/ `ICCAD_NO_REFINE=1`；
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
