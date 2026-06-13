# ICCAD 2026 FloorSet — Session Context

## Claude 對話框規範
- 聊天室語句**盡量精簡**、用**繁體中文**。

## 🚨 範式轉移（最重要，先讀）

**這題是 reconstruction（還原 baseline 原圖），不是 floorplan optimization。**

- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel)`：gap=0 ∧ V_rel=0 → **Cost=1.0**（理論最小）。我們「找最佳解」永遠 HPWL_gap>0 → Cost>1；還原原圖才能 gap≈0。
- 真天花板 ~1.1（`fp_sol` verbatim = **1.1079**）。組員 **1.0322 是 oracle**（讀本地 validation label，hidden test 退回 fallback → 不適用）；legit 上限 ~1.62（無 label portfolio）。
- 訓練資料的 `fp_sol` = ground truth (w,h,x,y)。我們的無監督 loss（只看 HPWL+overlap）完全沒用它 → GNN 學的是「散開短連線」而非「原圖」。組員 v10-v12 各種 ML/clue 也全 <1% → per-block local ML 在這題很弱。
- **現況：我們 1.3843 已反超組員所有 legit 版本（~14.6%）。所有 optimization lever（knob / 後處理 / order / global-packer）已枯竭 → 唯一剩餘 headroom = reconstruction**（用 connectivity + constraints 反推原佈局），需要的不是更強 optimizer。

## 評分公式（2026-05-23 確認）

- **Cost**（per case）= `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, R^0.3)`
  - 不可行 = 10.0；feasible 上限 9.999999；gap 從下方 clamp 到 0（贏過 baseline 無額外獎勵）
  - `V_rel = (V_boundary + V_grouping + V_mib) / N_soft`，`N_soft = boundary blocks + Σ(MIB-1) + Σ(Cluster-1)`
- **Total Score** = `Σ Cost[i]·exp(n_i/12) / Σ weight`
  - 權重 e^(n/12)：n=120→8.0%、n≥110 累計 ~53%（中小型 case 比舊版重很多）；總權重 ≈ 275418
- **RuntimeFactor** = `max(0.7, R^0.3)`，R 分母 = cross-submission median（未知，組員 ~11s 唯一參考）。本地 eval 強制 =1.0。**懲罰比 = (t1/t2)^0.3，與 median 無關** → **8-11s 安全帶，>13s 必虧**。

## 目前狀態

### 🏆 最佳：Total Score = 1.3843（M26, 39-prof, 2026-06-13）

`constructive.cpp`（C++ 建構式定框 placer，B 路線重寫組員架構）+ `optimizer_constructive.py`（portfolio wrapper）。**確定性**（無 randomness/限時 → run-to-run 一致，可精確 A/B）、100/100 feasible、8.78s/case。**proxy 自 M13 起 = oracle ceiling**（完美選擇，加 profile 全額 realize → selection 不再是瓶頸）。

### 單 profile 架構（5 階段，~0.16s/case）
1. **boundary-aspect dims**：LEFT/RIGHT-only aspect **2.50**、TOP/BOTTOM-only **0.40**（拉高 edge capacity 降 vBd，最高 ROI insight）
2. **MIB 形狀統一**（`apply_safe_mib_dims`）：master 相容→用 master；否則 movable ≤1% area→`sqrt(avg)` 方形。保 1% 硬約束 → vMb 145→0
3. **cluster 建構**：純 movable→複合 item（3 ordering×5 layout，key=`(fragments, boundary_bad, area, aspect)` 字典序，fragment/boundary 排 area 前）；mixed(preplaced+movable)→anchored（first-pass 貼 preplaced「牆」）
4. **定框 greedy packing**：試 4-5 個 outline frame（面積小優先），每 item boundary-aware 候選評分（`bbox_area + 0.10·anchor + ww·WIRE·wire + BP_W·boundary_miss`），ww base **×2000**（平坦盆地非尖峰）；layout_score 挑最佳 frame
5. **後處理**：compaction（M10）→ wire refinement（M9）→ HPWL push/slide/swap/jump（M14-16/24）

### Portfolio 層
平行跑 39 deterministic profile（env 旋鈕變體），用 **baseline-free proxy** 選最佳：
- proxy = `(area/Â + _RH·hpwl/hmin)·exp(2·vrel)`，Â=1.035·ΣblockArea，hmin=該 case 各 profile 最小 hpwl，**_RH=1.4**（補償 hmin/hbase≈1.3-1.4 對 hpwl 項的低估）
- ⚠️ **vrel 必須用 shapely 算**（wrapper `_proxy_metrics`），不可用 C++ union-find（1e-3 tol，34/100 案不一致 → 退到 1.6x）
- 下檔保護：無用 profile 不被選、不傷分（只花 runtime）

### 演進里程碑（deterministic A/B；M4 起累計 -38.5%）
M1 singles 3.62 → M2 cluster 2.35 → M4 +MIB/layout-key/wire×2000 1.82 → M5 anchored cluster 1.7045 → M6-8 portfolio(7→13) 1.5659 → M9 wire refinement 1.5375 → **M10 %.17g 精度修正 + compaction 1.4528** → M11 迭代 compaction 1.4502 → M12 40-prof 1.4371 → M13 narrow frame + _RH=1.4 1.4349 → M14 HPWL push(free single) 1.4253 → M15 boundary-axis slide 1.4236 → M16 same-size swap 1.4231 → M17 WIRE_TIEBREAK 1.4202 → M18 WIRE_BFS 1.4138 → M19 BFS_PIN 1.4105 → M20 ORDER_SWAP 1.4080 → M21 OS 組合(K=16) 1.3998 → M22 OS16 移植 1.3987 → M23 ORDER_MOVE 1.3983 → **M24 HPWL jump(跨障礙) 1.3862** → M25 池審計剪枝(56→38, runtime -27%, 分數不變) → **M26 GUIDE_MED(39-prof) 1.3843** → M27 global-packer 探測 = 死路。

## 🔑 戰略結論：所有 optimization lever 枯竭（M26-M27 兩次天花板探測）

1. **ordering / ML 整分支永久封卷**（M26 oracle-perm，`oracle_perm_probe.py` + `ICCAD_ORDER_FILE`）：注入**完美 fp_sol 排序**，placer 只多拿 **+0.002%（類內）/ +0.005%（全域）** → **瓶頸是 placer（greedy+compact+push），不是 pack order**。⇒ refinement pair-relocation / order-LNS / 監督式 ML ranking **全不值得做**。（對應舊 SA oracle-perm 3.27 天花板，現對 constructive 證實。）
2. **「更好的 packer」面封死**（M27 global-packer，`dbg_seqpair.py`）：把 greedy 佈局用 **sequence-pair** 全域重排 + 退火（RELAXED 樂觀上限：clusters 打散、preplaced 不釘、boundary 不計 → 真實只會更差）。greedy seed + 20k 退火 **0 改善**；shelf bad-seed 收斂到 greedy area 但 HPWL 1.49×（**拿不到 (area,HPWL) 聯合點**）；hard case ≤2% 樂觀且幾乎全是 trivial LB-compaction 假象。**根因：agap 與 hgap 耦合（wire-driven 花 area 換低 HPWL）+ cluster/preplaced 強迫 void = 結構性**，非 packing 品質 → B*-tree/SP/skyline 重寫不值得。（SP recovery 正解 = overlap-conditioned 邊，pairwise L/R/B/A 會循環。）
3. **唯一剩餘 headroom = reconstruction**（見下節「未來發展方向」）。

## 未來發展方向

> 所有 optimization lever 已枯竭（上節兩次天花板探測 + 下方死路 ledger）。剩餘方向依 ROI：

### 1. Reconstruction（唯一真正的 frontier，~1.1 理論上限）
從「optimization（壓 area/hpwl/violation）」轉成「還原 baseline 原圖」——gap≈0 才能逼近 Cost=1.0。需用 b2b/p2b connectivity + constraints 反推「原圖怎麼擺」。
- ⚠️ **明顯的子路徑都已封死**，reconstruction 需要全新想法，不是把現有 placer 餵更好的 order/start：
  - ML ranking / pack-order：M26 oracle-perm 證實 placer 是天花板（完美排序僅 +0.005%）
  - supervised MSE 對 fp_sol 絕對位置：v2 失敗（ill-posed 一對多，疊在中間）
  - 全域重排（sequence-pair / B*-tree）：M27 證實 greedy 已在 (area,HPWL) 前緣
- 可能的新角度（未驗證）：(a) 研究組員純演算法怎麼用 constraint+connectivity 反推佈局；(b) 預測**相對結構 / 接觸圖**（哪些 block 該 abut）而非絕對位置，再用 placer 實現該接觸關係；(c) 整合 *global* tiling/structure 訊號（contact graph、outline aspect）而非 per-block local feature。

### 2. per-block free-aspect（次要未測小 lever）
soft block 在 ±1% area 內選 aspect，packing 時一起搜。與 global packer 正交。期望低（dims 已被 boundary-aspect + LR/TB profile 大半覆蓋），但是唯一還沒碰過的 placer-層 knob。

### 3. 精度 / 數值（持續遵守）
任何新加的、會被 shapely 評分的幾何輸出都要保持精確 abutment + `%.17g`（見 Gotchas）。

## 死路 ledger（勿重試）

- **boundary aspect port 到舊 SA**（2.50/0.40）：3.3258→3.4255 退步 3%（我們 skyline ≠ 組員 shelf，tall block 成 cliff 害後續找位）
- **preplaced-aligned frame**（攻 case 89）：greedy pack 不下 tighter width，case 89 結構性無解；贏案全被現有 profile 蓋過（零貢獻）
- **cluster-rigid pack/slide**：cluster 無 slack（100 案僅 1 個能動）+ FP 剛體平移破壞精確 abutment → shapely 虛假 fragment（**M10 精度牆，任何移動 cluster 成員的後處理都會撞**）
- **violating boundary 修復**：202 violating = 123 cluster + 45 preplaced + **34 single 全 BLOCKED** → 真值 0 個可修（`dbg_vio_stats.py`）。**residual vBd 只能靠 packing 階段擺對**
- **per-frame compaction + csc 重估 frame**：csc 固定 hw 跨 outline 失準（拿 vCl 換 vBd），單 base 退步；跨 layout 選擇是 wrapper shapely proxy 的工作
- **reframe**（compact 後實測 bbox seed frame 重跑，`ICCAD_REFRAME`）：base frame loop 已挑最佳 aspect，pass2 複製 pass1；與 portfolio aspect 多樣性結構性冗餘（code 保留 gated off）
- **env knob 軸**：WIRE_MULT 4/6、LR+W、ANCHOR 0.30、ultra-narrow frame、WT/BFS/NORM/PIN knob 組合、CLUSTER_ORD、OM×tight — 全 ≤0.063%
- **OS K>16 / OM K 組合**：K=16 飽和（高權重案 jump 紅利已拿光），更大 K 只撿中型案渣且 runtime 不划算
- **compaction 方向偏好 / pack 向 connectivity 重心**：compact_layout 已對稱試 4 單向+8 兩步組合由 csc 仲裁（方向偏好是嚴格子集）；wire 項已是 placed+guide 動態重心
- **runtime 候補 om16/os24/os32**：M24 jump 吃掉賣點（96/66/89 headline 已被超越），自身 wall 主導每案 → 懲罰比永遠不划算
- **BP_WEIGHT**：30000→1M 完全無變化（不是 penalty 太低，是無可行 bp=0 位置 / frame 邊≠bbox 邊）
- **試更多 frame**：all-frames → 2.42（layout_score 150000·bv 在大池 overshoot）。4-5 frame 最佳
- **wire ×50000** → 1.93 反彈；**wire_order**（wire 當第一鍵）→ vBd 390
- **ML：shape / perm ranking**：oracle 上限實驗 = BL packer 是天花板（perm+SA 3.27、shape only 3.42）→ 都被 placer 架構 cap 住

## 殘留 case（純 optimization 已榨乾）
89 **1.7936**（最高，preplaced boundary 撐壞 outline）、85 1.6091、62 1.5227、88 1.4354、79 1.4121、87 1.3505、91 1.3481、66 1.3981。硬 case（89/85/62）= preplaced boundary 幾何強迫，需 packer pack tight（M13+M27 證實非 packer 能解）。

## 環境 & 指令

- **主程式**：`constructive.cpp` + `optimizer_constructive.py`（舊 SA `optimizer_claude.cpp/.py` 僅 fallback）
- **Conda**：`C:\Users\Nordra\.conda\envs\iccadv\python.exe`；**Compiler**：`C:\msys64\ucrt64\bin\g++.exe`
- 組員參考碼：`C:\Users\Nordra\Downloads\teammate_iccad_study\`

```powershell
# 編譯（Bash 工具寫 .exe 會失敗，務必用 PowerShell）
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive.exe constructive.cpp

# 官方 portfolio eval（~3 分鐘，確定性）
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_constructive.py 2>&1 | Select-Object -Last 12

# 快速單 profile A/B（~70 秒，乾淨確定性，與官方吻合）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_constructive.py 2>&1 | Select-Object -First 12
```

## Gotchas
- **PowerShell 用 `;` 或 `if ($?){...}` 連接，不能用 `&&`**；Bash 工具寫 .exe 會失敗（sandbox）
- **輸出必須 `%.17g`**（非 %.10f）：否則精確 abutment 被捨入成虛假 shapely cluster fragment（M10，~144 假 fragment/100 案，是 -6.3% 單一最大 lever）。新增任何被 shapely 評分的幾何都要遵守；查不明 grouping violation 先疑精度
- **proxy 選擇用 shapely vrel**（`_proxy_metrics`），不可用 C++ METRICS vrel
- **compaction 選候選用 true-cost csc** `(area+hw·hpwl)·exp(2·(bv+gf)/nsoft)`，不可用 layout_score（後者 boundary 權 150000 ≫ grouping 6500，會拿 vCl 換 vBd）
- 新增重 profile 前先查它自己的 per-case cpu（每案 wall ≈ 最重 profile 的 max 項，非池總量）

## env 旋鈕（`constructive.cpp`）
- 預設：`ICCAD_BP_WEIGHT`=30000、`ICCAD_WIRE_MULT`=×1、`ICCAD_ANCHOR_W`=0.10、`ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT`
- 關後處理：`NO_COMPACT` / `NO_REFINE` / `NO_PUSH`（關 M14-16+24）/ `NO_BND_PUSH`（退 M14）/ `NO_SWAP`（退 M15）/ `NO_JUMP`（退 M16）；`PUSH_PASSES=N`、`COMPACT_ITERS=N`、`REFINE_ITERS=N`
- pack-order 軸：`WIRE_TIEBREAK`、`WIRE_BFS`、`BFS_PIN`、`ORDER_SWAP=K`、`ORDER_MOVE=K`、`GUIDE_MED`（M26 ship）
- 死路（code 保留 gated off，勿重掃）：`BFS_NORM`、`CLUSTER_ORD=1/2`、`REFRAME`
- 離線探測（永不 ship）：`ORDER_FILE=path` + `ORDER_GLOBAL=1`（oracle-perm）
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 退單 base profile（1.7045）

## 工具
- `analyze_constructive.py`（單 profile per-case breakdown，~30s）
- `profile_vs_portfolio.py KEY=VAL`（新 profile 候選逐案比 portfolio 算 oracle-min 增益，>0.05% 才加）
- `rh_sweep.py`（真值快取 + 掃 _RH/proxy）、`portfolio_ceiling.py`（oracle 天花板）、`proxy_dbg.py`、`profile_audit.py`（M25 池審計：win tally/LOO/cpu）
- `dbg_area.py`（area density）、`dbg_boundary.py` / `dbg_vio_stats.py`（violation 分類）、`dbg_compact.py` / `dbg_compact_cmp.py`、`dbg_hpwl_push.py`（push 原型）
- 負結果工具（保留）：`oracle_perm_probe.py`（M26 ordering 探測）、`dbg_seqpair.py`（M27 global-packer 探測）

## 檔案結構（要點）
- `constructive.cpp` 🏆 — placer，含 M9-M26 全部行為（見 env 旋鈕）+ METRICS stderr
- `optimizer_constructive.py` 🏆 — 39-prof portfolio + shapely proxy(_RH=1.4)；M25 剪 18 條（`[M25-pruned]` 註解可復原）
- `optimizer_claude.cpp/.py/.exe` — 舊 SA，僅 fallback
- `floorplan_gnn.pth` — v1 GNN（unsupervised，僅舊路線）；`gnn_training.md` — ML 文件
- `iccad2026contest/iccad2026_evaluate.py` — 評估腳本

## ML（已 park，詳見 `gnn_training.md`）
- **v1**（unsupervised, 2 層 GCN）：退役，曾配舊 SA 到 3.3258
- **v2**（supervised MSE on `fp_sol`）：**失敗**（ill-posed 一對多：同 X 多個合法 Y，MSE 收斂到平均=疊在中間，unsup_cost 暴衝 47M）。`floorplan_gnn_v2.pth` **勿 commit**
- **v3**（pairwise ranking）：sanity rank_acc 0.53→0.58 訊號弱；且 oracle-perm 證實 placer 是天花板 → **ML 路線 park**
- ⚠️ 本環境**禁止跑訓練**（要訓練複製到 GPU 環境）

## 舊 SA 路線（fallback only，`optimizer_claude`）
峰值 portfolio 8-profile 3.0625（contest-shape proxy + GNN-hint + W_BOUNDARY=100）。constructive 已全面超越，僅作 constructive 失敗時 fallback。`analyze_violations.py` / `check_viols.py` 不可跑（用 `viol_breakdown.py`）；`pack_cluster_anchored` 在 code 但不被呼叫。
