# ICCAD 2026 FloorSet — Session Context

## Claude 對話框規範
- 聊天室語句**盡量精簡**、用**繁體中文**。

## 🚨 範式轉移（最重要，先讀）

**這題是 reconstruction（還原 baseline 原圖），不是 floorplan optimization。**

- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel)`：gap=0 ∧ V_rel=0 → **Cost=1.0**（理論最小）。我們「找最佳解」永遠 HPWL_gap>0 → Cost>1；還原原圖才能 gap≈0。
- 真天花板 ~1.1（`fp_sol` verbatim = **1.1079**）。組員 **1.0322 是 oracle**（讀本地 validation label，hidden test 退回 fallback → 不適用）；legit 上限 ~1.62（無 label portfolio）。
- 訓練資料的 `fp_sol` = ground truth (w,h,x,y)。我們的無監督 loss（只看 HPWL+overlap）完全沒用它 → GNN 學的是「散開短連線」而非「原圖」。組員 v10-v12 各種 ML/clue 也全 <1% → per-block local ML 在這題很弱。
- **現況：我們 1.3279（M36）已反超組員所有 legit 版本（~18.0%）。packer/order lever 自 M26-27 枯竭，但 free-aspect 軸 M29 重開 → ratio/boundary/cluster-uniform/cluster-per-member/anchored 五子軸全 ship（M29-M35），M36 再榨乾 anchored 殘渣（寬 ratio + ungate boundary，兩個「預判死」全翻盤）。OS16×free 確定 runtime-死。剩餘 headroom = reconstruction**（RED，M29 從零 builder 8.22 證偽），需要的是重寫 placer 架構而非更強 optimizer。

## 評分公式（2026-05-23 確認）

- **Cost**（per case）= `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, R^0.3)`
  - 不可行 = 10.0；feasible 上限 9.999999；gap 從下方 clamp 到 0（贏過 baseline 無額外獎勵）
  - `V_rel = (V_boundary + V_grouping + V_mib) / N_soft`，`N_soft = boundary blocks + Σ(MIB-1) + Σ(Cluster-1)`
- **Total Score** = `Σ Cost[i]·exp(n_i/12) / Σ weight`
  - 權重 e^(n/12)：n=120→8.0%、n≥110 累計 ~53%（中小型 case 比舊版重很多）；總權重 ≈ 275418
- **RuntimeFactor** = `max(0.7, R^0.3)`，R 分母 = cross-submission median（未知，組員 ~11s 唯一參考）。本地 eval 強制 =1.0。**懲罰比 = (t1/t2)^0.3，與 median 無關** → **8-11s 安全帶，>13s 必虧**。

## 目前狀態

### 🏆 最佳：Total Score = 1.3279（M36 anchored 殘渣 wide-ratio + ungate-boundary, 39-prof, 2026-06-21；avg 9.86s, est_wall 7.64s OS16-bound 不變, 100/100 feasible）

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
M1 singles 3.62 → M2 cluster 2.35 → M4 +MIB/layout-key/wire×2000 1.82 → M5 anchored cluster 1.7045 → M6-8 portfolio(7→13) 1.5659 → M9 wire refinement 1.5375 → **M10 %.17g 精度修正 + compaction 1.4528** → M11 迭代 compaction 1.4502 → M12 40-prof 1.4371 → M13 narrow frame + _RH=1.4 1.4349 → M14 HPWL push(free single) 1.4253 → M15 boundary-axis slide 1.4236 → M16 same-size swap 1.4231 → M17 WIRE_TIEBREAK 1.4202 → M18 WIRE_BFS 1.4138 → M19 BFS_PIN 1.4105 → M20 ORDER_SWAP 1.4080 → M21 OS 組合(K=16) 1.3998 → M22 OS16 移植 1.3987 → M23 ORDER_MOVE 1.3983 → **M24 HPWL jump(跨障礙) 1.3862** → M25 池審計剪枝(56→38, runtime -27%, 分數不變) → **M26 GUIDE_MED(39-prof) 1.3843** → M27 global-packer 探測 = 死路 → M28 reconstruction 天花板 probe（GREEN）→ M29 tree decoder 拆解（X=B\*-tree 100% 精確）+ 從零 builder = 死路 → **M29 per-block free-aspect ship（base+wire+GM+tight 共 4 profile，43-prof）1.3787**（M26 以來首個動分數 lever，−0.41%，攻動 n=118 大 case；10.16s） → **M30 free+PIN combo family（剪 9、加 4 free，38-prof）1.3694**（−0.67% 本 session；`free_pin_tight`/`free_gm_pin` 為全池 LOO 前二，case 95 n=116→**1.1967**、98 n=119→**1.3323**；OS16 仍不可剪、牆結構性；~9.2s clean）→ M31 aspect-ratio + proxy/_RH 雙軸再探=死路（分數不變 1.3694）→ **M32 decoupled boundary-aspect（高 LR + TB 留 default 0.40；加 4 解耦、剪 3 冗餘，39-prof）1.3667**（−0.20% 本 session；硬案 85 n=106 1.6091→**1.5364**(LR4.5)、71 n=92→**1.2760**(LR3.0)；per-block FREE_BOUNDARY=0.000% 死路 → boundary-aspect win 須 UNIFORM profile 非 per-block；est_wall 8.03s、feasible 100/100）→ **M33 cluster-member aspect（新 `ICCAD_CLUSTER_ASPECT` 旋鈕；6 輪 probe，加 10 cluster profile、剪 10 冗餘含 base，39-prof）1.3390**（**−2.03% 本 session，自早期 constructive 以來最大單 session 增益**；標準 sweep 2.0 僅 +0.14%，但 **ca-ratio stack tight+free+pin 放大效應、最優遠寬**：`ca3.0+free+pin+tight` +1.105%（共振峰、LOO +0.857% 全池最大），破長年最難 case 89 n=110 1.7292→**1.6142** + 98/90/87；est_wall ~7.6s、feasible 100/100；free-aspect 三軸 ratio/boundary/cluster 全 ship）→ **M34 per-member cluster aspect（實作 `ICCAD_FREE_CLUSTER` + `ICCAD_FREE_CLUSTER_RATIOS` 旋鈕；2 互補 profile fc_pin_tight/fc_gm_pin、剪 4 冗餘 LR，37-prof）1.3325**（−0.49% 本 session；per-member 在硬案**勝過 M33 uniform**：最難 case 89 1.5954→**1.5640**、82 1.420→**1.363**、80 1.490→**1.402** + 88/79/66/83/92/70；ratio 共振峰 **4.0**（5.0 退步），**單一 profile 即讓同 cluster 成員異質 tall(0.333)+wide(4.0)**，取代 M33 分開的 ca0.4/ca3.0；2 profile 為全池 LOO 前二（13+6 wins）；est_wall 7.54s、avg 9.21s、feasible 100/100；free-aspect 四子軸 ratio/boundary/cluster-uniform/cluster-per-member 全 ship）→ **M35 anchored-cluster free-aspect（新 `ICCAD_FREE_ANCHORED` 旋鈕；mixed cluster 的 movable 成員在 wall-attach first-pass 搜 aspect；加 1 profile fc_anchored_pin_tight，38-prof）1.3313**（−0.090% 本 session；**原預測會撞 M32 失敗模式（packing-score 仲裁、無 cluster layout-key）卻 ship**——anchored score 的 `bp`+keep-connected 項翻盤：stacked on fc_pin_tight +0.111% oracle-min、proxy 全額 realize，破長年最難 case 89 1.5640→**1.5538**、65 1.7325→**1.6904**、52 1.4071→**1.3613** + 78；standalone 0.000%（放大-only，同 M33/M34）；GM 變體冗餘 win⊂PIN；avg 9.30s、est_wall OS16-bound 不變、feasible 100/100；free-aspect 第五子軸）→ **M36 anchored 殘渣（新 `ICCAD_FREE_ANCHORED_RATIOS`/`ICCAD_FREE_ANCHORED_BND` 旋鈕；加 2 profile fc_anchored_bnd_pin_tight/fc_anchored_wide_pin_tight、剪 1 冗餘 tall_asp5_all，39-prof）1.3279**（−0.255% 本 session、2.8× M35；**兩個「預判死」全翻盤**：① 寬 anchored ratio（重用 FREE_RATIOS max 2.0 → wide 0.333..4.0、共振 4.0；per-frame 但 anchored 成員少 → **wall-free** 4.3s，異於 M31）② ungate boundary（`FREE_ANCHORED_BND`，M32 FREE_BOUNDARY 類比再被 anchored score 的 `bp`+keep-connected 推翻）；**互補**：ungated 破硬案 97 n=118 1.2279→**1.1988**（全池 LOO 第一 +0.00235）、gated 破 88 n=109 1.4115→**1.3852**（ungated 反輸此案）；M35 5-ratio gated 仍貢獻不被剪；avg 9.86s、est_wall 7.64s、feasible 100/100。**新死路：OS16×free +0.630% oracle-min 但 48s/n=120（4× OS16），增益=OS swap loop 內 per-frame free 搜尋、與 4× runtime 結構耦合（OS16+PIN+tight 無 free=+0.001%）→ 入 OS24/32/om16 shelf**）。

## 🔑 戰略結論：packer/order lever 枯竭，但 aspect lever M29 重開（M26-M29）

> ⚠️ M29 更新：「所有 lever 枯竭」過頭了——**free-aspect（aspect 軸）一直沒試**，M29 一試就 ship（1.3843→1.3814）。封死的是 **packer 架構（M27）與 pack-order/ML（M26）**，不含 block 形狀。下方 1-2 點仍成立；reconstruction（第 3 點）由 M28 GREEN 經 M29 拆解後下修 YELLOW（decoder 懂了但 tree 測試時不可得）。

1. **ordering / ML 整分支永久封卷**（M26 oracle-perm，`oracle_perm_probe.py` + `ICCAD_ORDER_FILE`）：注入**完美 fp_sol 排序**，placer 只多拿 **+0.002%（類內）/ +0.005%（全域）** → **瓶頸是 placer（greedy+compact+push），不是 pack order**。⇒ refinement pair-relocation / order-LNS / 監督式 ML ranking **全不值得做**。（對應舊 SA oracle-perm 3.27 天花板，現對 constructive 證實。）
2. **「更好的 packer」面封死**（M27 global-packer，`dbg_seqpair.py`）：把 greedy 佈局用 **sequence-pair** 全域重排 + 退火（RELAXED 樂觀上限：clusters 打散、preplaced 不釘、boundary 不計 → 真實只會更差）。greedy seed + 20k 退火 **0 改善**；shelf bad-seed 收斂到 greedy area 但 HPWL 1.49×（**拿不到 (area,HPWL) 聯合點**）；hard case ≤2% 樂觀且幾乎全是 trivial LB-compaction 假象。**根因：agap 與 hgap 耦合（wire-driven 花 area 換低 HPWL）+ cluster/preplaced 強迫 void = 結構性**，非 packing 品質 → B*-tree/SP/skyline 重寫不值得。（SP recovery 正解 = overlap-conditioned 邊，pairwise L/R/B/A 會循環。）
3. **reconstruction headroom = +0.276（M28 GREEN），但 M29 下修為 YELLOW**：M29（`tree_decode_probe.py`）完全拆解 tree→geometry——`tree_sol` 是 **B\*-tree**，fp_sol 的 **X = B\*-tree X 規則 100% 精確**（n=24..117，不需位置 label），fp_sol 的 Y = 固定 X 的垂直重力壓實但**順序非 tree 拓樸序**；renderer（exact-X + 依序 gravity）給對的 Y 序 → **完全重現 fp_sol**（拓樸序面積 1.13×~1.80×，大 case 更糟）。⇒ 重建 = 「給定精確 X 下的 1D Y 排序」，**但測試時無 tree → 須從 connectivity 同時建出 (X 結構 + Y 序 + aspect + 約束) = 我們已解的 placement**。M29 從零 connectivity builder（`tree_build_probe.py`）= **死路**（8.22 vs 1.38、0/100，爛在 HPWL+violations，非面積）→ 有競爭力的 builder = 重寫 constructive.cpp。詳見下節。

## 未來發展方向

> 所有 optimization lever 已枯竭（上節兩次天花板探測 + 下方死路 ledger）。剩餘方向依 ROI：

### 1. Reconstruction — **M28 GREEN → M29 拆解後下修 YELLOW**（decoder 已懂、從零 builder 死路）

**M28 reconstruction ceiling probe（`reconstruct_probe.py`，offline，2026-06-14）**：輸出 fp_sol verbatim 過官方 evaluator，拆解天花板。決定性結果：
- **Oracle = 1.1079（再確認）；headroom = 1.3843−1.1079 = +0.2764**，是 M13 以來所有 optimization 增益總和（1.4231→1.3843=0.039）的 **~7×**。
- **headroom 100% 在 quality（hpwl+area），violation 已贏**：our quality factor 1.274（hgap 0.328 + agap 0.221）vs oracle 1.000；our Vrel 0.040 **低於** oracle 0.050。⇒ reconstruction = 「擺出和原圖一樣緊（低 area）又短（低 HPWL）的 feasible 佈局」，**violation 面不用碰**。
- **1.1 floor 是原圖自身的 boundary V_rel，焊死在資料裡**（90/100 案 oracle boundary>0）→ 真天花板 ~1.106 不可再降；但與我們無關（我們已在 floor 之下做 violation）。
- **100/100 feasible、block 全矩形**（max polygon 頂點=5）→ fp_sol 可用矩形 verbatim 輸出 → reconstruction 是**位置/品質還原**（非 metric-only 的難級跳）。
- **headroom 高度集中**：top-15 案（全 n=103–120 + 硬案 89/85）佔 68.9% → tractable，非散落。
- **目標表徵 = 插入式 slicing tree**：`tree_sol` shape `(n−1)×3` = (anchor=col0, 新block=col1, cut_dir=col2∈{0,1})，**112/112 layouts 結構乾淨**（col1=非seed的排列、col0=已置 block）。原圖是從這棵樹生成的 slicing floorplan。
- **connectivity 帶結構訊號但非決定性**：tree-attached pair 有 **38.2%** 是 b2b 直接相連 vs random **19.9%**（~2×）→ 可學，但**一對多是真的**（62% 非直接邊，無法用單純 min-cut 唯一還原）。

**為何 M27 沒封死這條**：M27 只「重排 greedy 的固定矩形」（SP-repack，固定 dims、打散 cluster）；**從沒建過 netlist-driven 的 slicing-tree placer + free aspect**。原圖的 (area,HPWL) 優勢來自**全域 slicing 結構 + 非方形 aspect**，是我們 skyline-greedy 架構結構性產不出的點 → 換 placer **架構**（slicing-tree constructive）才碰得到，與死路 ledger 的「SP/B*-tree 重排 greedy」正交。

**⚠️ 已封死的子路徑（勿重試）**：ML ranking/pack-order（M26 oracle-perm +0.005%）；supervised MSE 對絕對位置（v2 ill-posed）；SP/B*-tree 重排 greedy 矩形（M27）；**從零 connectivity tree builder（M29，`tree_build_probe.py`，8.22 vs 1.38）——重寫 placer 不值得，要改善只在 constructive.cpp 內加 lever**。

**M29 結果（2026-06-14，`tree_decode_probe.py` + `tree_build_probe.py`，OFFLINE）**：
1. ✅ **decoder 解開**：X = B\*-tree X 規則 100% 精確、label-free；renderer（exact-X + 依序 gravity）忠實（給對 Y 序 → 完全重現 fp_sol）。fp_sol ≠ 自身 tree 的 B\*-tree pack（right-child 可落 parent 下方）→ Y 是 post-pack 垂直壓實、順序非拓樸。
2. ❌ **從零 builder = 死路**：connectivity-greedy + free-aspect（含 frame-bounded）8.22 vs 1.38、0/100；爛在 HPWL+violations（忽略 boundary/grouping/MIB）。有競爭力 = 重寫 constructive.cpp 全部機制。
3. ⚠️ **reconstruction 下修 YELLOW**：tree（精確 X 的來源）測試時不可得 → 仍須從 connectivity 建 placement。剩餘兩條，依 ROI：

### 1b. connectivity→tree ML map（探索，低優先；M29 後保留為唯一「reconstruction-specific」路）
學「連通性 → B\*-tree 結構」以還原 X（Y 序仍須另解）。但 M28 訊號弱（**2×、一對多**），且本環境**禁訓練**（需 GPU 環境，見 ML 節）。在 free-aspect 拿到正向訊號前不投入。

### 2. per-block free-aspect — ✅ **M29 SHIPPED 1.3787 → M30 free+PIN family 1.3694**
M28「headroom 全在 quality」+ M29「原圖 X=B\*-tree、用非方形 aspect」⇒ free-aspect 是**唯一未試、不需 tree 的 GREEN lever**。實作 = `constructive.cpp` 候選評分裡為 **single interior movable block** 在 ±1% area（精確同面積）內搜 aspect（`FREE_RATIOS={1.0,1.5,0.6667,2.0,0.5}`），gated `ICCAD_FREE_ASPECT`，自包含分支 + `continue` → FREE_ASPECT=0 bit-identical。
- **全域 `ICCAD_SOFT_ASPECT`（先試的粗版）只動小 case**（1.6/0.62 各 ~0.11% oracle-min，wContr≈0）→ 留旋鈕但未進 portfolio。
- **per-candidate（`ICCAD_FREE_ASPECT=1`）攻動大 case**：base+free oracle-min **0.184%**，case 97 n=118 1.269→1.253（wContr 0.109%）、76/61/57/67…。
- **M29：4 個 free profile 進 portfolio（43-prof）→ 1.3843→1.3787（−0.41%）**：`free_aspect`、`free_aspect_wire`、`free_gm_wt_wire`、`free_tight_wire`。
- **M30 SHIPPED（2026-06-15）：1.3787→1.3694（−0.67% 本 session）、~9.2s clean、100/100、38-prof**。三步（詳見 `optimizer_constructive.py` 註解 + `audit_M30.txt`）：
  1. **Phase B 審計剪枝（`profile_audit.py`，score-neutral）**：剪 5 個 wins==0 ∧ |LOO|<1e-12（`wire_hi`/`tall_anclo`/`narrow_wire_anc`/`os_bfs_wt_wire`/`gm_bfs_wt_wire`，標 `[M30-pruned]`）。⚠️ **核心假設「free 偷走 OS16 wins → 剪 OS16 降牆」= False**：OS16（~21s max）牆主宰全部存活、各賺各案（5/2/8 wins）→ est_wall 7.39 不變，剪枝只降 contention 不降牆。
  2. **加 4 個 wall-safe free combo（`profile_vs_portfolio.py`，>0.05% 才加）**：r1 `free_pin_wt_wire`（+0.233%，case 98/95/89）、`free_gm_tight_wire`（+0.134%，case 95/65/40）；r2 `free_pin_tight_wire`（+0.405% 增量，case 95 n=116 **1.2400→1.1967**、73/87）、`free_gm_pin_wt_wire`（+0.284%，case 98 n=119 **1.3704→1.3323**）。**r2 兩個是全池 LOO 前二（+0.0041/+0.0028）**。皆單趟 packing ~9s max → 藏 OS16 天花板下、牆-安全。
  3. **round-2 審計再剪 4（被 free 強版蓋過、score-neutral）**：`anc_lo`/`area_lean`/`tight_wire`/`wtb_wire`（標 `[M30r2-pruned]`）。
  - **飽和訊號（停手點）**：r3 全合併 `free+GM+PIN+tight` 僅 **+0.006%** → free+PIN family 榨乾。`free+tall`（只贏 89、與 PIN 重疊）、`free+narrow`（0%）已死。
  - ~~**proxy 滑移觀察（潛在未來 lever）**~~：**M31 證偽**——「case 95 選 1.2400 vs pool 1.2329」是 M30 中間態，最終池 `free_pin_tight` 已把 95 拉到 1.1967 且 proxy 正選 → 38-prof oracle-min==proxy(_RH=1.4)==1.3694，選擇器完美、零 headroom（見死路 ledger）。
- **M32 SHIPPED（2026-06-19）：1.3694→1.3667（−0.20% 本 session）、est_wall 8.03s、100/100、39-prof**。lever = **decoupled UNIFORM boundary aspect**（既有 `ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT` 旋鈕，零 C++ code）。三步：
  1. **Phase 0 pre-probe（`profile_vs_portfolio`，全域 LR/TB sweep）**：既有 `aspect_*` profile 全把高 LR 配低 TB（兩軸一起壓），漏掉 TB 想留 default 0.40 的案。解耦掃出 **LR=4.5 +0.186%**（硬案 85 n=106 1.6091→**1.5364**，共振非單調：3.5→0.101%、4.0→0.009%、4.5→0.186%、5.0→0.014%）、LR=3.0（case 71→1.2760）、TB=0.8（49）、TB=0.667（67/52）。
  2. **加 4 解耦 profile（`{LR:4.5}`/`{LR:3.0}`/`{TB:0.8}`/`{TB:0.667}`，全 keeper wins>0∧LOO>0）+ 審計剪 3 被吸收的 coupled profile（`tall_asp5_wire`/`wtb_tall_wire`/`bfs_tall_wire`，標 `[M32-pruned]`，score-neutral）**。
  3. **飽和訊號（停手點）**：stack/fine-ratio（LR4.5+wire、LR4.6、TB0.8+wire…）全 <0.05% bar；boundary-aspect lever 榨乾。
- ❌ **per-block FREE_BOUNDARY = 死路（M32，0.000% oracle-min）**：greedy 局部 area 項偏好窄塊（寬 LR 塊右擴大 bbox）→ 與 edge-capacity 需要的「統一變扁」反向 → win 須 **UNIFORM**（profile 級）aspect，非 per-block。code 已 revert（見死路 ledger）。
- **剩餘 free-aspect 方向**：① ~~finer/wider `FREE_RATIOS`~~ **M31 死路**；② ~~boundary per-block~~ **M32 死路**（uniform 版已 ship）；③ ~~cluster 成員 uniform~~ **M33 ship**；④ ~~cluster 成員 per-member~~ **M34 ship**；⑤ ~~anchored(mixed) cluster 成員 per-member~~ **M35 ship**（原判 <0.05% 死路，實際 +0.111%；M32 失敗模式預測被 anchored score 的 `bp`+keep-connected 項推翻）。**僅剩殘渣（live follow-up，先驗未知）**：FREE_ANCHORED 目前重用 `FREE_RATIOS`（max 2.0），M34 cluster-member 共振峰在 **4.0** → **anchored 寬 ratio 未試**（但 per-frame ∝ frames → 須查 wall）；+OS/+GM 其他 stack 未試。

### ✅ cluster-member free-aspect = M33 uniform + M34 per-member SHIPPED（1.3667→1.3390→1.3325）

- **M33 uniform `ICCAD_CLUSTER_ASPECT`**（單一 ratio reshape 全部純 movable interior cluster 成員）：standalone sweep 訊號弱（2.0 +0.14%），真正大 win 是 **stack tight+FREE+PIN 放大、ratio 遠寬**——`ca3.0+free+pin+tight` +1.105%（共振峰，LOO +0.857%）。見 memory `[[m33-cluster-aspect]]`。
- **M34 per-member `ICCAD_FREE_CLUSTER`**（各成員獨立搜 aspect，用既有 cluster layout-key 仲裁）：**per-member 在硬案勝過 uniform**——同案直接比贏 case 89 1.5954→**1.5640**、82→**1.363**、80→**1.402** + 88/79/66/83/92/70。ratio 共振峰 **4.0**（5.0 退步），build-time 搜尋（非 per-frame）→ widen 免抬牆，**單一 profile 即讓同 cluster 成員異質 tall(0.333)+wide(4.0)**，取代 M33 分開的 ca0.4/ca3.0。2 profile（fc_pin_tight +0.466%、fc_gm_pin +0.414%）為全池 LOO 前二。見 memory `[[m34-free-cluster]]`。

### 📋 NEXT SESSION 計畫：free-aspect + anchored 殘渣全 ship（M36）→ RED reconstruction / 收束

**所有 wall-safe in-env optimization lever 枯竭**：packer/order（M26-27 天花板）+ free-aspect 五子軸（ratio M31 / boundary M32 / cluster-uniform M33 / cluster-per-member M34 / anchored M35）+ anchored 殘渣（寬 ratio + ungate boundary，M36）全 ship 或死路。proxy/_RH 完美（零漏分）。
⚠️ **M35→M36 教訓（二度應驗）：「預測會死」≠ 死**——anchored 的三個殘渣（M35 base、M36 寬 ratio、M36 ungate boundary）都被類比 M32 FREE_BOUNDARY 判死，全翻盤（anchored score 的 `bp`+keep-connected 項使仲裁不同）。**bounded probe 值得做，別只憑類比預判死路。** 但 **runtime-coupled 的就是死**（M36 OS16×free +0.630% 但 48s，與 4× runtime 不可拆，見死路 ledger）。
**剩餘方向依 ROI**：
1. ~~**anchored free-aspect 殘渣**~~ **M36 SHIPPED 1.3279**（寬 ratio + ungate boundary 互補，破 97/88）。**僅剩 runtime-coupled 殘渣（OS×free）已證死**。anchored 軸真正榨乾。
2. **RED reconstruction**（重寫 placer 為 netlist-driven slicing-tree + free aspect）：headroom 仍大（oracle 1.1079，剩 +0.224），但 **M29 從零 connectivity builder 已證偽 8.22 vs 1.38（0/100）** → 須重寫 constructive.cpp 全部約束機制（boundary/grouping/MIB/精度），非加 lever。ROI 極低、風險極高。
3. **收束本題**：1.3279 領先組員所有 legit 版本 ~18.0%，runtime 安全（avg 9.86s）。若無新 insight，合理停點。
- ⚠️ **eval 實際 ~15-17 分鐘**（100 案 serial，大案 n>110 每案 ~20-26s）；profile_audit 全 re-collect ~15 分。background 用 harness `run_in_background`（完成通知，比 detached 可靠）；`> file 2>&1` 在 PowerShell 對 native exe 會印 cosmetic `NativeCommandError`（無害，進度照跑）。

### 3. 精度 / 數值（持續遵守）
任何新加的、會被 shapely 評分的幾何輸出都要保持精確 abutment + `%.17g`（見 Gotchas）。

## 死路 ledger（勿重試）

- **boundary aspect port 到舊 SA**（2.50/0.40）：3.3258→3.4255 退步 3%（我們 skyline ≠ 組員 shelf，tall block 成 cliff 害後續找位）
- **preplaced-aligned frame**（攻 case 89）：greedy pack 不下 tighter width，case 89 結構性無解；贏案全被現有 profile 蓋過（零貢獻）
- **cluster-rigid pack/slide**：cluster 無 slack（100 案僅 1 個能動）+ FP 剛體平移破壞精確 abutment → shapely 虛假 fragment（**M10 精度牆，任何移動 cluster 成員的後處理都會撞**）
- **violating boundary 修復**：202 violating = 123 cluster + 45 preplaced + **34 single 全 BLOCKED** → 真值 0 個可修（`dbg_vio_stats.py`）。**residual vBd 只能靠 packing 階段擺對**
- **per-frame compaction + csc 重估 frame**：csc 固定 hw 跨 outline 失準（拿 vCl 換 vBd），單 base 退步；跨 layout 選擇是 wrapper shapely proxy 的工作
- **reframe**（compact 後實測 bbox seed frame 重跑，`ICCAD_REFRAME`）：base frame loop 已挑最佳 aspect，pass2 複製 pass1；與 portfolio aspect 多樣性結構性冗餘（code 保留 gated off）
- **env knob 軸**：WIRE_MULT 4/6、LR+W、ANCHOR 0.30、ultra-narrow frame、WT/BFS/NORM/PIN knob 組合、CLUSTER_ORD、OM×tight — 全 ≤0.063%（⚠️ **M32 例外**：「LR+W ≤0.063%」是因把 LR 綁 W / 低 TB；**pure decoupled LR**（高 LR + TB 留 default 0.40）一直沒試 → LR=4.5 +0.186% 攻硬案 85，已 ship）
- **per-block boundary free-aspect（`ICCAD_FREE_BOUNDARY`，M32）**：0.000% oracle-min。greedy 局部 score 的 area 項偏好窄塊（寬 LR 塊往內擴 bbox）→ 與 edge-capacity 要的「統一變扁」**反向**、per-block 主動選錯 aspect ⇒ boundary-aspect win 結構性須 **UNIFORM**（profile 級設定），per-block 死。code 已 revert（0.000% 不留 dead code）。uniform decoupled 版 = M32 SHIPPED
- **cluster aspect 飽和邊界（M33，勿重掃）**：uniform `ICCAD_CLUSTER_ASPECT` standalone sweep 訊號弱（2.0=+0.14%、0.8/1.5 <0.03%）——真正 win 須 **stack tight+FREE+PIN 放大**。寬 ratio 共振峰在 **3.0**（+1.105%）；3.5/4.0/5.0 僅 0.3-0.4%、adjacent 2.5/3.25 <0.12%（重疊 ca3.0）；tall 0.4>0.6、0.33≈0.10%；wide-GM-tight 0.042% 死。已加 ca{2.0,0.6,1.25,0.4,3.0}×{plain/tight/GM} 過 0.05% 者。
- **cluster aspect per-member 飽和邊界（M34，勿重掃）**：per-member `ICCAD_FREE_CLUSTER`（build-time 搜尋、widen 免抬牆）共振峰在 **4.0**（PIN stack +0.466%）；**5.0 退步**（+0.367%）、3.0 +0.348%、narrow(max2.0) 僅 +0.143%、standalone +0.135%。已加 fc_pin_tight + fc_gm_pin（GM 互補，+0.414%），剪 4 冗餘 LR（aspect_xhi/lr30/asp7_wire/asp5_all，被 per-member 寬 ratio 蓋過硬案）。**per-member 勝 uniform 於硬案**（89/82/80）。殘渣 anchored-cluster 成員 = **M35 SHIPPED +0.111%**（非死路；原判 <0.05% 被推翻，見里程碑/`ICCAD_FREE_ANCHORED`）；**anchored 寬-ratio + ungate boundary = M36 SHIPPED**（+0.255%,二度翻盤）；**anchored +OS = runtime-死**（OS×free 48s,見死路 ledger）→ anchored 軸全榨乾
- **OS K>16 / OM K 組合**：K=16 飽和（高權重案 jump 紅利已拿光），更大 K 只撿中型案渣且 runtime 不划算
- **compaction 方向偏好 / pack 向 connectivity 重心**：compact_layout 已對稱試 4 單向+8 兩步組合由 csc 仲裁（方向偏好是嚴格子集）；wire 項已是 placed+guide 動態重心
- **runtime 候補 om16/os24/os32**：M24 jump 吃掉賣點（96/66/89 headline 已被超越），自身 wall 主導每案 → 懲罰比永遠不划算
- **OS16 × free-aspect 家族（M36，勿重試）**：+0.630% oracle-min（贏 23 案含高權重 90/88/73/82）但 **48s/n=120（4× OS16 的 12s）**。增益本質 = OS swap loop（120 packs/frame）內跑 per-frame free 搜尋（FREE_ASPECT/FREE_ANCHORED × swap count）→ 與 4× runtime **結構耦合不可拆**:**OS16+PIN+tight 無 free = +0.001%**（便宜部分池子既有 OS16 profile 已覆蓋）。入 OS24/32/om16 runtime shelf。GM 變體:gated+GM +0.008%、ungated+GM +0.067%（唯一 disjoint case 64 < bar）→ PIN-only（同 M35）
- **BP_WEIGHT**：30000→1M 完全無變化（不是 penalty 太低，是無可行 bp=0 位置 / frame 邊≠bbox 邊）
- **試更多 frame**：all-frames → 2.42（layout_score 150000·bv 在大池 overshoot）。4-5 frame 最佳
- **wire ×50000** → 1.93 反彈；**wire_order**（wire 當第一鍵）→ vBd 390
- **ML：shape / perm ranking**：oracle 上限實驗 = BL packer 是天花板（perm+SA 3.27、shape only 3.42）→ 都被 placer 架構 cap 住
- **FREE_RATIOS 加寬**（M31，2026-06-19）：補 2.5/0.4+1.25/0.8（9-wide）→ 整池 oracle 僅 **−0.044%**（1.3694→1.3688）；單 profile `profile_vs_portfolio` 探針亮眼（free_pin_tight +0.31%、free_gm_pin +0.086% 含 hard case 66 1.3936→1.3429）但**幾乎全是與其他 7 個 free profile 的重疊**（rh_sweep 整池 oracle 才是真相）。且抬牆 max 21→**29.63s**（6 案>21s 超 OS16），score 增益與 runtime 全集中在 n>110 大案（free 搜尋 ∝ n²）→ **結構耦合不可解**。7-ratio 子集更小且仍抬牆 → 全軸死。proxy 在 wide 池仍完美（1.3689 vs oracle 1.3688）
- **proxy/_RH 選擇**（M31 再確認）：38-prof 池 oracle-min==proxy(_RH=1.4)==**1.3694**，選擇器零漏分（M30「case 95 滑移」是中間態，已被最終池 `free_pin_tight` 修正 95→1.1967）→ _RH 非 live lever，要降分須降 oracle 本身（剩 cluster/boundary free-aspect 或 RED reconstruction）

## 殘留 case（純 optimization 已榨乾）
89 **~1.554**（M35 anchored 從 M34 的 1.5640 再降，仍最高；preplaced boundary 撐壞 outline）、85 ~1.536（M32 LR4.5）、65 **~1.690**（M35 從 1.7325）、62 1.5227、88 **~1.385**（M36 從 1.412）、97 **~1.199**（M36 ungate-boundary 從 1.228）、82 ~1.363（M34）、52 **~1.361**（M35 從 1.4071）、79 ~1.247（M34）、66 ~1.270（M34）、91 1.3481。硬 case（89/85/62）= preplaced boundary 幾何強迫；**M33→M36 證實成員形狀（cluster-member + anchored-member aspect，含寬 ratio + ungate boundary）+ tight frame 能顯著鬆動 89/82/88/97/79/66/65/52**（非 packer 架構問題，是成員形狀；anchored 成員貼牆形狀也能優化，推翻「自由度低」預判）。

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
- **aspect 軸（M29/M30）**：`ICCAD_FREE_ASPECT=1`（per-block：single interior movable 在 ±1% area 搜 aspect，**M30 ship 1.3694**，8 free profile 進 portfolio：base/+wire/+GM-wt-wire/+tight-wire（M29）+ **+PIN-wt-wire/+GM-tight-wire/+PIN-tight-wire/+GM-PIN-wt-wire（M30，後二 `free_pin_tight`/`free_gm_pin` 為全池 LOO 前二）**）；`ICCAD_SOFT_ASPECT=r`（全域 interior aspect，預設 1.0；粗版只動小 case，未進 portfolio）
- **cluster aspect 軸 uniform（M33）**：`ICCAD_CLUSTER_ASPECT=r`（reshape 純 movable INTERIOR cluster 成員 w/h，預設 1.0=方形=bit-identical；套在 `solve()` `apply_safe_mib_dims()` 後、`make_group_item` 前）。**M33 ship 1.3390**，cluster profile 進 portfolio。⚠️ 標準 uniform sweep 訊號弱（2.0 +0.14%），真正大 win 是 **stack tight+FREE+PIN 放大、ratio 遠寬**：`ca3.0+free+pin+tight`=+1.105%（共振峰）、`ca2.0+...+tight`=+0.582%、tall `ca0.6/0.4`、GM 變體
- **cluster aspect 軸 per-member（M34）**：`ICCAD_FREE_CLUSTER=1`（各 INTERIOR 純 movable 成員獨立在 `FREE_CLUSTER_RATIOS` 搜 aspect，在 `make_group_item` 內座標下降、用既有 cluster layout-key 仲裁；預設 off=bit-identical）+ `ICCAD_FREE_CLUSTER_RATIOS=r1,r2,...`（搜尋集，預設 `1.0,1.5,0.6667,2.0,0.5`，env 覆寫）。**M34 ship 1.3325**，2 profile（fc_pin_tight/fc_gm_pin，用 wide set `0.333..4.0`）為全池 LOO 前二。**build-time 搜尋（非 per-frame）→ widen 免抬牆**；ratio 共振峰 **4.0**（5.0 退步）；**per-member 勝 uniform 於硬案**（89/82/80）。⚠️ 仲裁用 cluster layout-key（非 packing greedy-area）避 M32 per-block boundary 失敗模式
- **aspect 軸 anchored per-member（M35）**：`ICCAD_FREE_ANCHORED=1`（mixed=preplaced+movable cluster 的 movable 成員，在 `pack_in_frame` wall-attach first-pass 搜 aspect over `FREE_RATIOS`；**per-frame 像 FREE_ASPECT、只 commit `out[]` 不寫 `dims[]`**（避跨 frame 污染）；gate `boundary==0 && mib==0 && !is_fixed`；預設 off=bit-identical，sentinel ratio -1）。**M35 ship 1.3313**，1 profile `fc_anchored_pin_tight`。⚠️ 仲裁用 packing greedy score（M32 FREE_BOUNDARY 同模式）**卻有效**——anchored score 的 `bp`+keep-connected 項翻盤；破 case 89/65/52。standalone 0.000%、須 stack `fc_pin_tight` 放大；GM 變體冗餘（win⊂PIN）
- **aspect 軸 anchored 殘渣（M36）**：`ICCAD_FREE_ANCHORED_RATIOS=r1,r2,...`（FREE_ANCHORED 搜尋集,預設 == `FREE_RATIOS`=bit-identical,env 覆寫;**per-frame → 加寬須查 wall**,但 anchored 成員少 → wide `0.333..4.0`（共振 4.0）wall-free 4.3s）+ `ICCAD_FREE_ANCHORED_BND=1`（也讓 boundary anchored 成員搜 aspect,預設 0=gate=bit-identical）。**M36 ship 1.3279**,2 profile:`fc_anchored_bnd_pin_tight`（ungated,全池 LOO 第一 +0.00235,破硬案 97→**1.1988**）+ `fc_anchored_wide_pin_tight`（gated wide,破 88→**1.3852**,互補；ungated 反輸 88）。⚠️ ungate boundary 原判 M32 FREE_BOUNDARY 死路,被 anchored score 的 `bp`+keep-connected 推翻（同 M35 模式,二度應驗）
- 死路（code 保留 gated off，勿重掃）：`BFS_NORM`、`CLUSTER_ORD=1/2`、`REFRAME`
- 離線探測（永不 ship）：`ORDER_FILE=path` + `ORDER_GLOBAL=1`（oracle-perm）
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 退單 base profile（1.7045）

## 工具
- `analyze_constructive.py`（單 profile per-case breakdown，~30s）
- `profile_vs_portfolio.py KEY=VAL`（新 profile 候選逐案比 portfolio 算 oracle-min 增益，>0.05% 才加）
- `rh_sweep.py`（真值快取 + 掃 _RH/proxy）、`portfolio_ceiling.py`（oracle 天花板）、`proxy_dbg.py`、`profile_audit.py`（M25 池審計：win tally/LOO/cpu）
- `dbg_area.py`（area density）、`dbg_boundary.py` / `dbg_vio_stats.py`（violation 分類）、`dbg_compact.py` / `dbg_compact_cmp.py`、`dbg_hpwl_push.py`（push 原型）
- 探測工具（保留）：`oracle_perm_probe.py`（M26 ordering 天花板）、`dbg_seqpair.py`（M27 global-packer 天花板）、`reconstruct_probe.py`（M28 reconstruction 天花板拆解：oracle 1.1079、headroom 100% quality；快取 `reconstruct_probe_cache.json`）、`tree_decode_probe.py`（M29 decoder：tree_sol=B\*-tree，X 規則 100% 精確、Y 序是品質 lever；讀訓練 `floorset_lite/*.th`）、`tree_build_probe.py`（M29 從零 connectivity builder = 死路 8.22）

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
