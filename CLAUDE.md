# ICCAD 2026 FloorSet — Session Context

## Claude 對話框規範
- 聊天室語句**盡量精簡**、用**繁體中文**。

## 🚨 範式轉移（最重要，先讀）

**這題是 reconstruction（還原 baseline 原圖），不是 floorplan optimization。**

- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel)`：gap=0 ∧ V_rel=0 → **Cost=1.0**（理論最小）。「找最佳解」永遠 HPWL_gap>0 → Cost>1；還原原圖才能 gap≈0。
- 真天花板 ~1.1（`fp_sol` verbatim = **1.1079**，headroom 100% 在 quality、violation 已贏）。組員 **1.0322 是 oracle**（讀本地 validation label，hidden test 退 fallback → 不適用）；legit 上限 ~1.62。
- 訓練 `fp_sol` = ground truth (w,h,x,y)，但無監督 loss 沒用它 → per-block local ML 在這題很弱（組員 v10-v12 全 <1%）。
- **現況：M41（2026-06-25）開全新 RuntimeFactor 軸（與 quality 正交，被本地 eval 強制 RF=1.0 藏住）；M42（2026-06-26）打第二槍**。本地分 1.3269→**1.3277**（RF=1.0 fiction，+0.06%、之後恆定）：M41 砍 6 隻 swap profiles（OS16×3/OS8×2/OM8，大案 18-20s 純 runtime 死重）→ avg 9.89→5.90s；**M42 再砍 21 隻「不贏任何 n>100 案」的 build profiles（per-big-n 冗餘，非全域 LOO）→ avg 5.90→4.73s（−20%）、big-case wall n=120 砍半 15.6→8.0s、quality BIT-IDENTICAL（local 仍 1.3277、20 案全 median-independent WIN）**、投影 real 疊加 M41 共 **~−22%**（@ M=11、robust 跨 M∈[6,20]）、100/100 feasible。**quality 軸仍 converged**（packer/order M26-27 + free-aspect 六子軸 M29-M37 + reconstruction M40-RED 全探畢）；RF 軸唯一前提是「官方真套 RF」（否則只損 +0.06%，極不對稱賭注）。逐版細節 git log + memory（`[[m42-runtime-2nd-order]]`、`[[m41-runtime-factor]]`…`[[m29-tree-decoder]]`）。

## 評分公式（2026-05-23 確認）

- **Cost**（per case）= `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, R^0.3)`
  - 不可行 = 10.0；feasible 上限 9.999999；gap 從下方 clamp 到 0（贏過 baseline 無額外獎勵）
  - `V_rel = (V_boundary + V_grouping + V_mib) / N_soft`，`N_soft = boundary blocks + Σ(MIB-1) + Σ(Cluster-1)`
- **Total Score** = `Σ Cost[i]·exp(n_i/12) / Σ weight`
  - 權重 e^(n/12)：n=120→8.0%、n≥110 累計 ~53%（中小型 case 比舊版重很多）；總權重 ≈ 275418
- **RuntimeFactor** = `max(0.7, R^0.3)`（逐案，`evaluate.py:552`），R 分母 = cross-submission median（未知，組員 ~11s 唯一參考）。本地 eval **強制 =1.0**（`:924-940`）→ **M1-M37 每次 A/B 對此項全盲，1.3269 是 RF=1.0 fiction**。**懲罰比 = (t1/t2)^0.3，與 median 無關**（→ 逐案可本地判定）。**M41 反轉：RF 不只是約束、是 lever**——`cost∝t^0.3`，砍大案 wall 是 median-independent 的 real-score 增益（同案 `Q_cap/Q_full < (t_full/t_cap)^0.3` 即贏；大案品質結構卡死 → 幾乎恆贏）。見 `[[m41-runtime-factor]]`、`rf_score_model.py`。

## 目前狀態

### 🏆 最佳：local 1.3277 / 投影 real ~−22%（M41+M42 RuntimeFactor lever, two-tier adaptive, 2026-06-26；**avg 4.73s**, 100/100 feasible）

M41+M42 default-on（`ICCAD_ADAPTIVE_POOL=1`）**兩階砍 profiles**：M41 砍 swap（全案）、M42 砍 21 隻 build 冗餘（`block_count>100`，`ICCAD_ADAPTIVE_FREE_N=100`）；**`ICCAD_ADAPTIVE_POOL=0` 還原 full 40-prof = quality-best 1.3269**（M37, avg 9.89s）、`ICCAD_ADAPTIVE_FREE_N=9999` 退 M41-only。⚠️ 官方 local eval 顯示 **1.3277**（RF=1.0 fiction，看似退步；M42 對 local 零再損、bit-identical）——真增益在 real RF 項，本地永遠看不到（見 RuntimeFactor 節）。

`constructive.cpp`（C++ 建構式定框 placer，B 路線重寫組員架構）+ `optimizer_constructive.py`（portfolio wrapper）。**確定性**（無 randomness/限時 → run-to-run 一致，可精確 A/B）、8.78s/case 單 profile。**proxy 自 M13 起 = oracle ceiling**（完美選擇，加 profile 全額 realize → selection 不再是瓶頸）。

### 單 profile 架構（5 階段，~0.16s/case）
1. **boundary-aspect dims**：LEFT/RIGHT-only aspect **2.50**、TOP/BOTTOM-only **0.40**（拉高 edge capacity 降 vBd，最高 ROI insight）
2. **MIB 形狀統一**（`apply_safe_mib_dims`）：master 相容→用 master；否則 movable ≤1% area→`sqrt(avg)` 方形。保 1% 硬約束 → vMb 145→0
3. **cluster 建構**：純 movable→複合 item（3 ordering×5 layout，key=`(fragments, boundary_bad, area, aspect)` 字典序）；mixed(preplaced+movable)→anchored（first-pass 貼 preplaced「牆」）
4. **定框 greedy packing**：試 4-5 個 outline frame（面積小優先），每 item boundary-aware 候選評分（`bbox_area + 0.10·anchor + ww·WIRE·wire + BP_W·boundary_miss`），ww base **×2000**；layout_score 挑最佳 frame
5. **後處理**：compaction（M10）→ wire refinement（M9）→ HPWL push/slide/swap/jump（M14-16/24）

### Portfolio 層
平行跑 40 deterministic profile（env 旋鈕變體），用 **baseline-free proxy** 選最佳：
- proxy = `(area/Â + _RH·hpwl/hmin)·exp(2·vrel)`，Â=1.035·ΣblockArea，hmin=該 case 各 profile 最小 hpwl，**_RH=1.4**（補償 hmin/hbase≈1.3-1.4 對 hpwl 項的低估）
- ⚠️ **vrel 必須用 shapely 算**（wrapper `_proxy_metrics`），不可用 C++ union-find（1e-3 tol，34/100 案不一致）
- 下檔保護：無用 profile 不被選、不傷分（只花 runtime）

### 演進里程碑（deterministic A/B；M4 起累計 −38.5%）
M1 singles 3.62 → M2 cluster 2.35 → M4 +MIB/layout-key/wire×2000 1.82 → M5 anchored 1.7045 → M6-8 portfolio 1.5659 → M9 wire refine 1.5375 → **M10 %.17g + compaction 1.4528** → M12 40-prof 1.4371 → M13 narrow frame + _RH=1.4 1.4349 → M14-16 HPWL push 1.4231 → M17-23 pack-order 1.3983 → **M24 HPWL jump 1.3862** → M25 審計剪枝(38-prof) → **M26 GUIDE_MED 1.3843** → M27 global-packer 死路 → M28 reconstruction ceiling(GREEN) → **M29-M37 free-aspect 六子軸 1.3843→1.3269**（每子軸見下方 env 旋鈕 + memory）→ M38/M39 殘渣收束 → **M40 reconstruction RED-confirmed**（quality 軸到此 converged）→ **M41 RuntimeFactor lever（正交新軸）：砍 6 隻 swap profiles，local 1.3269→1.3277(+0.06%)、avg 9.89→5.90s(−40%)、投影 real ~−12%** → **M42 RF 二階：砍 21 隻 build 冗餘(n>100, per-big-n)，local 1.3277 bit-identical、avg 5.90→4.73s(−20%)、big-case wall 砍半、投影 real 疊加共 ~−22%**。

## 🔑 戰略結論：quality 三面天花板皆探畢；**RF 軸（M41+M42）為新開的正交維度、二階已 ship**

1. **ordering / ML 永久封卷**（M26 oracle-perm，`oracle_perm_probe.py`）：注入完美 fp_sol 排序，placer 只多拿 +0.002%（類內）/ +0.005%（全域）→ 瓶頸是 **placer**（greedy+compact+push）非 pack order。⇒ refinement pair-relocation / order-LNS / 監督式 ML ranking 全不值得。
2. **更好的 packer 封死**（M27 global-packer，`dbg_seqpair.py`）：greedy 已在 (area,HPWL) frontier；agap 與 hgap **結構耦合**（wire-driven 花 area 換低 HPWL）+ cluster/preplaced 強迫 void → B*-tree/SP/skyline 重寫不值得。
3. **reconstruction RED-confirmed**（M28 GREEN→M29 YELLOW→**M40 RED**）：headroom +0.219（oracle 1.1079，100% 在 quality，top-15 大案佔 68.9%）但 X 結構不可從 connectivity 還原（M40 Spearman 0.009）+ Y 序需 label（M40 deterministic +159% vs oracle-Y）→ 重寫 slicing placer 不可行。詳見死路 ledger + `[[m40-reconstruction-red]]`。
4. **RuntimeFactor 軸 OPEN（M41+M42）**：以上三點全是 **quality**；計分式還有 `max(0.7,R^0.3)` 一項，本地 eval 強制 =1.0 而被全程忽略。`cost∝t^0.3` + 大案 quality 卡死 + 大案佔 60% 權重 → 砍大案 wall = median-independent real-score 增益。已 ship **兩槍**：**M41** 砍 swap（avg 9.89→5.90s / 投影 real ~−12%）；**M42** 砍 21 隻「不贏任何 n>100 案」的 build 冗餘 profile（per-big-n，**非全域 LOO**；`rf_score_model.py` M42 區塊量化）→ big-case wall n=120 15.6→8.0s、avg 5.90→4.73s、quality **bit-identical**（local 仍 1.3277、20 案全 median-independent WIN）、投影 real 再 −11%（疊加共 −22% @ M=11）。**殘留三階**：multi-tier（n>110 再砍 #8/#12/#26/#27）僅 ~2-3%/10 案、且大案多樣性壓到 8 隻（overfit 風險）→ 未 ship。見 `[[m42-runtime-2nd-order]]`、`[[m41-runtime-factor]]`。

## 未來發展方向（M42 後）

> quality 軸 converged；**RF 軸 M41+M42 已 ship 兩槍**（砍 swap + 砍 build 冗餘）。依 ROI：

1. **RF 軸三階（marginal、overfit 風險，預設不追）**：M42 後大案 wall 由 13 隻 kept build winner 的 `max / sum-cores` 設底（n=120 ~8s，已近底）。multi-tier（n>110 再砍 #8/#12/#26/#27 等非-n>110-winner）僅再 ~2-3%/10 案、且把大案多樣性壓到 8 隻 → hidden-test overfit 風險；`rf_score_model.py` 亦可掃更低 T（90/95）但增益遞減。除非確認官方 median 極低，ROI 不值。
2. **submission hardening（不追分但對後續輪次實用）**：
   - **proxy generalization**：`_RH=1.4` 由本地 validation sweep——hidden test 分佈不同時是否仍 oracle-min？`rh_sweep.py` 看 1.3-1.6 平台寬度（窄=overfit 風險）。
   - **feasibility 保證**：M42 後仍 100/100；portfolio 全 fail 退 `python_sa_solve` fallback——確認 fallback feasible。M42 大案剩 13 profile、仍遠多於 1。
   - **RF/median 不確定性**：投影用組員 ~11s 錨 + 掃 M∈[6,25]；若官方 median 極高（大家都慢）則我們已近 0.7 floor、增益更大；若極低則增益縮但**不為負**（RF 對 t 單調）。最壞（官方忽略 RF）只損 +0.06% local（M42 不再加損）。
3. **若要再純-quality 追分**：須非 lever 的全新角度（新 constraint 結構洞察 / 新表徵），非既有六子軸延伸——目前沒有。

### 精度 / 數值（持續遵守）
任何新加的、會被 shapely 評分的幾何輸出都要保持精確 abutment + `%.17g`（見 Gotchas）。

## 死路 ledger（勿重試）

- **boundary aspect port 到舊 SA**（2.50/0.40）：3.3258→3.4255 退步（skyline ≠ 組員 shelf，tall block 成 cliff）
- **preplaced-aligned frame**（攻 case 89）：greedy 不下 tighter width，case 89 結構性無解
- **cluster-rigid pack/slide**：cluster 無 slack（100 案僅 1 能動）+ 剛體平移破壞 abutment → shapely 假 fragment（**M10 精度牆，任何移動 cluster 成員的後處理都撞**）
- **violating boundary 修復**：202 violating 真值 0 可修（`dbg_vio_stats.py`：123 cluster+45 preplaced+34 single 全 BLOCKED）→ residual vBd 只能 packing 階段擺對
- **per-frame compaction + csc 重估 frame**：csc 固定 hw 跨 outline 失準（拿 vCl 換 vBd）
- **reframe**（`ICCAD_REFRAME`）：與 portfolio aspect 多樣性結構性冗餘（code 保留 gated off）
- **env knob 軸**：WIRE_MULT 4/6、ANCHOR 0.30、ultra-narrow frame、WT/BFS/NORM/PIN 組合、CLUSTER_ORD、OM×tight 全 ≤0.063%（⚠️ M32 例外：pure decoupled LR=4.5 +0.186% 已 ship）
- **per-block FREE_BOUNDARY（M32）**：0.000%——greedy 局部 area 項偏好窄塊（與 edge-capacity 要的「統一變扁」反向）⇒ boundary-aspect win 須 **UNIFORM**（profile 級），per-block 死，code 已 revert。uniform 版 = M32 ship
- **cluster aspect uniform 飽和（M33，勿重掃）**：standalone 弱（2.0=+0.14%），win 須 stack tight+FREE+PIN，寬 ratio 共振峰 **3.0**（+1.105%）；已加 ca{2.0,0.6,1.25,0.4,3.0}
- **cluster per-member 飽和（M34，勿重掃）**：共振 **4.0**（PIN +0.466%）、5.0 退步、narrow(max2.0) 僅 +0.143%；已加 fc_pin_tight + fc_gm_pin
- **MIB-member 飽和（M37 ship + M38 收束，勿重掃）**：共振 ungated-wide **5.0**（89/61 ship）；tall 0.25→79 +0.027% below bar；M38 gated complement sweep 全 below bar（唯一 mover 79 gated-6.0 +0.033%）。per-case 選 ratio 不需 arbiter（portfolio 即選擇器）
- **FREE_CLUSTER boundary-ungate（`ICCAD_FREE_CLUSTER_BND`，M39，勿重掃）**：鏡像 M36 anchored-BND（layout-key 仲裁、build-time wall-free）。**LIVE 但 below bar**：92/100 案 / 391 members、layout-key 確實挑更好 boundary aspect（非 M32 純死），但贏案全低權重小/中案、硬案 89/97/85/82/88 任何 ratio 零移動 → 三 stack best +0.034% < bar → **revert**（3 行可復原：static 宣告 + gate `(boundary==0||FREE_CLUSTER_BND>0)` + env parse）
- **reconstruction 遞迴二分 slicing builder（`recon_slice_probe.py`，M40，勿重試）**：reconstruction **RED-confirmed**。M29 只證偽 greedy 插入 builder（8.22）；M40 補做正統 spectral 遞迴二分 slicing（Fiedler-median cut、region-aspect 切向）配 M29 `render_bstar` exact-X + oracle-Y。**雙閘各自獨立證死**（self-check `trueTree_oracleY` quality=1.0000 重現 fp_sol）：**Gate A（P1 X 還原）FAIL**——slicing-X+oracle-Y quality **2.05**（ceiling 1.0、我們 1.274）、movable X-order **Spearman 0.009**（pin-free 零相關）；**Gate B（P2 deterministic Y 序）FAIL**——即便給 **true tree X**，5 個 deterministic Y 序最佳 +**159%** vs oracle-Y（Y 需 label）。任一死即 RED。⚠️ 全集 100% 有 preplaced（mean 2.7、無 pure-movable）→ builder 須 obstacle-aware；但 Spearman 是 movable-only/pin-free，preplaced 救不了此 kill
- **OS K>16 / OM K 組合**：K=16 飽和；**OS16×free（M36）** +0.630% 但 48s/n=120（4× OS16，增益與 4× runtime 結構耦合，OS16+PIN+tight 無 free=+0.001%）；runtime 候補 om16/os24/os32 懲罰比不划算
- **compaction 方向偏好 / pack 向 connectivity 重心**：compact_layout 已對稱試 4 單向+8 兩步由 csc 仲裁；wire 已是動態重心
- **BP_WEIGHT 30000→1M 無變化**（無可行 bp=0 位置）；**試更多 frame** all-frames→2.42（4-5 frame 最佳）；**wire×50000**→1.93；**wire_order**→vBd 390
- **FREE_RATIOS 加寬（M31）**：整池 oracle 僅 −0.044% + 抬牆 21→29.6s（free 搜尋 ∝ n²，結構耦合）→ 全軸死
- **proxy/_RH（M31 再確認）**：oracle-min==proxy(_RH=1.4)==1.3269 零漏分 → 要降分須降 oracle 本身
- **ML shape / perm ranking**：oracle 上限實驗 = BL packer 是天花板（perm+SA 3.27、shape only 3.42）

## 殘留 case（純 optimization 已榨乾）
89 **~1.523**（M37，仍最高；preplaced boundary 撐壞 outline）、85 ~1.536（M32 LR4.5）、65 ~1.690（M35）、62 1.5227、88 ~1.385（M36）、97 ~1.199（M36 ungate-boundary）、82 ~1.363（M34）、52 ~1.361（M35）、61 ~1.303（M37）、79 ~1.247、66 ~1.270、91 1.3481。硬 case（89/85/62）= preplaced boundary 幾何強迫；M33-M37 證實成員形狀（cluster/anchored/MIB aspect + tight frame）能鬆動 89/82/88/97/79/66/65/61/52。

## 環境 & 指令

- **主程式**：`constructive.cpp` + `optimizer_constructive.py`（舊 SA `optimizer_claude.cpp/.py` 僅 fallback）
- **Conda**：`C:\Users\Nordra\.conda\envs\iccadv\python.exe`；**Compiler**：`C:\msys64\ucrt64\bin\g++.exe`
- 組員參考碼：`C:\Users\Nordra\Downloads\teammate_iccad_study\`
- ⚠️ **eval 實際 ~15-17 分鐘**（100 案 serial，大案 n>110 ~20-26s）；background 用 harness `run_in_background`（完成通知）；`> file 2>&1` 在 PowerShell 對 native exe 印 cosmetic `NativeCommandError`（無害）

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
- **輸出必須 `%.17g`**（非 %.10f）：否則精確 abutment 被捨入成虛假 shapely cluster fragment（M10，~144 假 fragment/100 案，−6.3% 單一最大 lever）。新增任何被 shapely 評分的幾何都要遵守；查不明 grouping violation 先疑精度
- **proxy 選擇用 shapely vrel**（`_proxy_metrics`），不可用 C++ METRICS vrel
- **compaction 選候選用 true-cost csc** `(area+hw·hpwl)·exp(2·(bv+gf)/nsoft)`，不可用 layout_score（後者 boundary 權 150000 ≫ grouping 6500，會拿 vCl 換 vBd）
- 新增重 profile 前先查它自己的 per-case cpu（每案 wall ≈ 最重 profile 的 max 項，非池總量）

## env 旋鈕（`constructive.cpp`）
- 預設：`ICCAD_BP_WEIGHT`=30000、`ICCAD_WIRE_MULT`=×1、`ICCAD_ANCHOR_W`=0.10、`ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT`（boundary aspect，預設 2.50/0.40）
- 關後處理：`NO_COMPACT`/`NO_REFINE`/`NO_PUSH`（關 M14-16+24）/`NO_BND_PUSH`/`NO_SWAP`/`NO_JUMP`；`PUSH_PASSES=N`、`COMPACT_ITERS=N`、`REFINE_ITERS=N`
- pack-order 軸：`WIRE_TIEBREAK`、`WIRE_BFS`、`BFS_PIN`、`ORDER_SWAP=K`、`ORDER_MOVE=K`、`GUIDE_MED`（M26 ship）
- **M41+M42 RF lever（wrapper 旋鈕，在 `optimizer_constructive.py solve()` 兩階 index-based filter，非 constructive.cpp）**：`ICCAD_ADAPTIVE_POOL`（預設 **1=on**）開啟兩階剪枝；`=0` 還原 full 40-prof（quality-best 1.3269）。**M41 階**：砍所有含 `ORDER_SWAP`/`ORDER_MOVE` 的 profiles（大案 18-20s 純 runtime 死重），`ICCAD_ADAPTIVE_N=K` 只砍 `block_count>K`（預設 0=全砍）。**M42 階**：再砍 `_BIG_REDUNDANT_IDX`（21 隻不贏任何 n>100 案的 build profile，模組級常數）當 `block_count>ICCAD_ADAPTIVE_FREE_N`（預設 **100**）；`=9999` 退 M41-only。⚠️ 改 `_PROFILES` 後須用 `rf_score_model.py` M42 區塊重算 `_BIG_REDUNDANT_IDX`。
- **free-aspect 六子軸**（全 gated、off=bit-identical；M29-M37 ship，per-case win 細節見 memory）：
  - `ICCAD_FREE_ASPECT=1`（M29/M30）：single interior movable 在 ±1% area 搜 aspect（`FREE_RATIOS={1.0,1.5,0.6667,2.0,0.5}`），per-candidate；8 free profile 進 portfolio（`free_pin_tight`/`free_gm_pin` 為全池 LOO 前二）
  - `ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT`（M32）：decoupled **uniform** boundary aspect（高 LR + TB 留 default 0.40）。⚠️ per-block FREE_BOUNDARY 死路 → 須 uniform
  - `ICCAD_CLUSTER_ASPECT=r`（M33）：uniform reshape 純 movable INTERIOR cluster 成員，套在 `apply_safe_mib_dims()` 後、`make_group_item` 前；共振 3.0
  - `ICCAD_FREE_CLUSTER=1` + `ICCAD_FREE_CLUSTER_RATIOS=r1,...`（M34，預設集 `1.0,1.5,0.6667,2.0,0.5`）：per-member 在 `make_group_item` 內搜 aspect、**cluster layout-key 仲裁**（非 greedy-area，避 M32 失敗）；build-time → widen 免抬牆，共振 4.0
  - `ICCAD_FREE_ANCHORED=1` + `ICCAD_FREE_ANCHORED_RATIOS=...` + `ICCAD_FREE_ANCHORED_BND=1`（M35/M36）：mixed cluster movable 成員在 `pack_in_frame` wall-attach 搜 aspect（per-frame、只 commit `out[]`）；gate `boundary==0&&mib==0&&!is_fixed`，BND=1 ungate boundary 成員
  - `ICCAD_MIB_ASPECT=r`（M37）：`apply_safe_mib_dims` 無-master 共享方形 reshape 成同面積共享矩形（保 MIB violation 0、all-interior gated）；共振 wide 5.0
  - `ICCAD_SOFT_ASPECT=r`：全域 interior aspect（粗版，未進 portfolio）
- 死路（code 保留 gated off / 未實作，勿重掃）：`BFS_NORM`、`CLUSTER_ORD=1/2`、`REFRAME`；`ICCAD_FREE_CLUSTER_BND`（M39 已 revert，3 行可復原）
- 離線探測（永不 ship）：`ORDER_FILE=path` + `ORDER_GLOBAL=1`（oracle-perm）
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 退單 base profile（1.7045）

## 工具
- `analyze_constructive.py`（單 profile per-case breakdown）、`profile_vs_portfolio.py KEY=VAL`（新 profile 逐案算 oracle-min 增益，>0.05% 才加）
- `rh_sweep.py`（真值快取 + 掃 _RH/proxy）、`portfolio_ceiling.py`（oracle 天花板）、`profile_audit.py`（M25 池審計：win tally/LOO/**per-profile cpu mean/max**，cache `audit_cache.pkl`）
- **`rf_score_model.py`（M41+M42 RF-aware 投影模型，永不 ship）**：載入 `audit_cache.pkl`，逐案 Q=proxy oracle-min、wall=`max(max_i t, Σt/cores)`，投影 real total=`Σw·Q·max(0.7,(t/M)^0.3)/Σw` 掃 median M。full-pool RF=1.0 重現 1.3269 為 sanity gate。逐案印 median-independent 判定 `Qcap/Qfull vs (tF/tC)^0.3`。**M42 區塊**：per-big-n winner tally → 候選冗餘集 `R = BUILD − {n>T winners}` → greedy 精煉（hmin 耦合用逐案 WIN check）→ 掃 T∈{100,105,110} 印投影增益 + 推薦 `_BIG_REDUNDANT_IDX`（改 `_PROFILES` 後重跑此區塊更新常數）
- `dbg_area.py`、`dbg_boundary.py` / `dbg_vio_stats.py`（violation 分類）、`dbg_compact*.py`、`dbg_hpwl_push.py`
- **離線天花板探測（永不 ship，保留為記錄）**：`oracle_perm_probe.py`（M26 ordering）、`dbg_seqpair.py`（M27 global-packer）、`reconstruct_probe.py`（M28：oracle 1.1079、headroom 100% quality；快取 `reconstruct_probe_cache.json`）、`tree_decode_probe.py`（M29 decoder：tree_sol=B\*-tree、X 規則 100% 精確、Y 序是品質 lever；讀訓練 `floorset_lite/*.th`）、`tree_build_probe.py`（M29 greedy builder 死路 8.22）、`recon_slice_probe.py`（M40 遞迴二分 slicing：雙閘死、reconstruction RED-confirmed）

## 檔案結構（要點）
- `constructive.cpp` 🏆 — placer，含 M9-M37 全部行為（見 env 旋鈕）+ METRICS stderr
- `optimizer_constructive.py` 🏆 — 40-prof portfolio + shapely proxy(_RH=1.4)；pruned profile 標 `[Mxx-pruned]` 註解可復原；**M41+M42 `solve()` default-on 兩階砍 profiles：swap（`ICCAD_ADAPTIVE_POOL`/`ICCAD_ADAPTIVE_N`）+ `_BIG_REDUNDANT_IDX` build 冗餘（`ICCAD_ADAPTIVE_FREE_N`，預設 100）**
- `optimizer_claude.cpp/.py/.exe` — 舊 SA，僅 fallback
- `floorplan_gnn.pth` — v1 GNN（unsupervised，僅舊路線）；`gnn_training.md` — ML 文件
- `iccad2026contest/iccad2026_evaluate.py` — 評估腳本

## ML（已 park，詳見 `gnn_training.md`）
- **v1**（unsupervised GCN）退役；**v2**（supervised MSE on `fp_sol`）失敗（ill-posed 一對多，`floorplan_gnn_v2.pth` 勿 commit）；**v3**（pairwise ranking）訊號弱 + oracle-perm 證實 placer 是天花板 → **ML 路線 park**
- ⚠️ 本環境**禁止跑訓練**（要訓練複製到 GPU 環境）

## 舊 SA 路線（fallback only，`optimizer_claude`）
峰值 8-profile 3.0625。constructive 已全面超越，僅作 fallback。`analyze_violations.py`/`check_viols.py` 不可跑（用 `viol_breakdown.py`）。
