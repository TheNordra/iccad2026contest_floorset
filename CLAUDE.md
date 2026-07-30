# ICCAD 2026 FloorSet — Session Context

> 本檔 2026-07-29 大幅精簡。**舊版完整敘述（每條 ledger 的逐案數字、probe 全文）留在 git commit `4e2eb42` 的 CLAUDE.md**，需要考古時 `git show 4e2eb42:CLAUDE.md`。證據本體在各 `MXX_REPORT.md` 與 memory。

## Claude 對話框規範
- 聊天室語句**盡量精簡**、用**繁體中文**。

## 🚨 先讀：這題是 reconstruction，不是 floorplan optimization

- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel)`：gap=0 ∧ V_rel=0 → Cost=1.0。「找更好的解」永遠 HPWL_gap>0 → Cost>1；**還原 baseline 原圖**才能 gap≈0。
- 真天花板 **1.1079**（`fp_sol` verbatim），headroom 100% 在 quality（violation 已贏）。組員 1.0322 是 **label oracle**（hidden test 不適用）；legit 上限 ~1.62。
- 但 **reconstruction 本身 RED**（M40）：X 結構無法從 connectivity 還原、Y 序需 label。⇒ 走「更好的 placer / 更聰明的 portfolio」而非「還原」。

### 現況一句話（2026-07-29）
- **local 最佳 = 1.3054（M71，已在 repo、regression_suite 7/7）**；**已上傳的 Beta tar 還是 1.3265（M67-G slim）** ⇒ 主要待辦 = 換件。
- 四大軸狀態：**quality 軸**——M26/M27/M40 三面天花板仍成立，但 **M71 證明 cluster 複合 item 的「候選集合／排序 key」是漏掉的軸**（−1.589% in-set / −4.04% OOS）；**RF 軸**——M41-M50 七槍 + tier-5 已 ship；**LP 軸**——offline GREEN（錨 1.2914）、in-window RED（M54/M62/組員 80cc719）；**ML 軸**——四種插入點全 RED（M52 生成、M56 selector、M68 seed、LP refinement）。
- 送件 hardening（M43/M48/M67-A~G）全部完成，Linux binary 雙邊逐位驗過。

## 評分公式（2026-05-23 確認）

- **Cost**（per case）= `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, R^0.3)`
  - 不可行 = 10.0；feasible 上限 9.999999；gap 從下方 clamp 到 0
  - `V_rel = (V_bnd + V_grp + V_mib) / N_soft`，`N_soft = boundary blocks + Σ(MIB−1) + Σ(Cluster−1)`
- **Total** = `Σ Cost[i]·exp(n_i/12) / Σ weight`（n=120 佔 8.0%、n≥110 累計 ~53%；總權重 ≈275418）
- **RuntimeFactor** = `max(0.7, R^0.3)` 逐案（`evaluate.py:552`），R 分母 = cross-submission median。
  - **alpha 校準（M67-E 定版）**：`M_i ≈ 3.161 × t_i^alpha`——錨的是 **alpha 那版（M10 廉價池）的逐案 runtime**（p50 0.673s），不是現行 shipped。48c 投影下我們**貼著 floor 邊緣**（加權 RF ≈0.73、mid band 0/40 觸底）⇒ runtime 在 48c 仍是活的分數項。組員「median ~11s」與 M67-D 的「8.2× 安全邊際」皆**作廢**。
  - 本地 eval **強制 RF=1.0**（`:924-940`）⇒ 所有 local 分數都是 RF=1.0 fiction，RF 增益本地永遠看不到。
  - **懲罰比 =（t1/t2)^0.3、與 median 無關** ⇒ 逐案可本地判定。**RF 是 lever 不只是約束**：`cost∝t^0.3`，砍大案 wall 是 median-independent 增益（`Q_cap/Q_full < (t_full/t_cap)^0.3` 即贏）。見 `[[m41-runtime-factor]]`、`rf_score_model.py`、`m67e_rf48.py`。

## 目前狀態

### 🏆 local **1.3054**（M71，2026-07-29，commit `0ff45f4`）
- 機制 = `make_group_item()` 純 movable cluster 複合 item 的**候選集合 + 排序 key**（見 ledger「M71」）。
- 驗證：旗標關 → `1.326473104916827` 0 movers；開 → `1.305389893450635`（與組員 json 逐位相同）、100/100 feasible、avg 1.75→**1.52s**（品質與 RF 雙贏）；OOS 240 案 `1.653329→1.586461`（**−4.04%**，是 in-set 的 2.5 倍）。
- ⚠️ **已量化缺口**：M42/M45/M49/M50 的常數是舊 placer 的 `audit_cache.pkl` 推的，M71 下 adaptive stack 的 in-set 品質稅 **+0.128% → +0.967%**（1.3054 vs `ADAPTIVE_POOL=0` 的 **1.2929**）。regen 是目前桌上最大的已知肉（~0.84%）。

### 📦 送件狀態（Beta deadline **2026-07-31 17:00 GMT+8**）
- **已上傳 = M67-G slim tar**（6 檔、md5 `06979081e47002a01d705cbfb9fc8f18`、Windows+Linux 雙邊逐位 `1.326473104916827`）。
- **M71 與 tier-5 都還沒進 tar** ⇒ 換件是最高優先。換件鏈 = `regression_suite.py`（7 項）→ `make_submission.py all` → `m67c_make_linux_bundle.py` + GPU 機 WSL `verify_final_tar.sh` → 確認 Drive 可覆蓋重傳。
- 格式硬規則：entry 必名 `op_wrapper.py`、禁絕對路徑、禁多餘 optimizer .py、禁未使用的大 binary（違反可能 DQ）。Beta 環境 = 每隊獨佔 **48c ICELAKE + A100**、真 hidden cases。
- alpha 成績：官方 **1.0286 = Rank 3**（送的是 M10 raw 1.4528 × cost-加權 RF 0.7081≈floor）；alpha 測資 = 本地 validation set 逐位相同。

### 引擎架構
`constructive.cpp`（C++ 建構式定框 placer）+ `optimizer_constructive.py`（portfolio wrapper）。**完全確定性**（無 randomness/限時）⇒ 可精確 A/B。單 profile ~0.16s/case。

單 profile 5 階段：
1. **boundary-aspect dims**：LEFT/RIGHT-only aspect 2.50、TOP/BOTTOM-only 0.40（拉高 edge capacity 降 vBd；最高 ROI insight）
2. **MIB 形狀統一**（`apply_safe_mib_dims`）：master 相容→用 master，否則 ≤1% area→`sqrt(avg)` 方形 ⇒ vMb 145→0
3. **cluster 建構**：純 movable→複合 item（M71 改良的候選/排序）；mixed→anchored（first-pass 貼 preplaced 牆）
4. **定框 greedy packing**：試 4-5 個 outline frame，每 item boundary-aware 候選評分（`bbox_area + 0.10·anchor + ww·WIRE·wire + BP_W·boundary_miss`，ww base ×2000），`layout_score` 挑最佳 frame
5. **後處理**：compaction → wire refinement → HPWL push/slide/swap/jump

Portfolio 層：平行跑 41 個 deterministic profile，用 **baseline-free proxy** 選最佳：
- proxy = `(area/Â + _RH·hpwl/hmin)·exp(2·vrel)`，Â=1.035·ΣblockArea，**_RH=1.4**
- **proxy 自 M13 起 = per-case oracle ceiling**（selection 不是瓶頸；加 profile 全額 realize）
- ⚠️ vrel **必須用 shapely**（wrapper `_proxy_metrics`），不可用 C++ union-find

里程碑一行：M1 3.62 → M10 `%.17g`+compaction 1.4528 → M13 proxy oracle 1.4349 → M24 HPWL jump 1.3862 → M29-M37 free-aspect 六子軸 1.3269 → M41-M50 RF 七槍（local 1.3285 = RF fiction、avg 9.89→1.49s）→ M51 wide-CLAMP 1.3265 → **M71 cluster-item 1.3054**。

## 🔑 戰略結論（哪些軸封了、哪些沒）

1. **ordering / ML ranking 封卷**（M26 oracle-perm）：注入完美 fp_sol 排序只多拿 +0.005% ⇒ 瓶頸是 placer 不是 order。誠實範圍：只測兩個 scalar sort key；anchored first-pass 內部順序的洞已由 M60 補診（前件空集）。
2. **packer 重寫封死**（M27）：greedy 已在 (area,HPWL) frontier、agap 與 hgap 結構耦合 ⇒ B*-tree/SP/skyline 不值得。誠實範圍：`dbg_seqpair.py` 是近似語意；M59/M61/M64 以官方 strict eval 攻鄰近軸亦全 RED。
3. **reconstruction RED**（M40）：X 從 connectivity 不可還原（Spearman 0.009）+ Y 序需 label（+159%）。
4. **RF 軸 = 已兌現的主力**（M41-M50 + M67-F tier-5）：alpha 實測 RF 0.7081≈floor 證明七槍全數兌現。
5. **🚨 但 quality 軸沒有 converged**（M40 的「converged」已被 M71 推翻）：M33-M39 掃的全是**成員 aspect**，沒人動過複合 item 的**候選集合與排序 key**。M71 −1.589% 就在那個洞裡。⇒ 找洞要找「從沒被參數化的結構決策」，不是再掃已知旋鈕的值。
6. **ML 四種插入點全 RED**：生成（M52 imitation 零容錯×零訊號）、selector（M56 winner case-idiosyncratic + proxy 已 oracle）、seed（M68 完美種子 vs portfolio 僅 +0.001%）、refinement（LP 系列 + M64）。重開唯一條件 = 取得 rival 1.29 的 legit 方法細節。

## 下一步（依 ROI）

1. **🚨 換件（deadline 前，唯一主線）**：把 M71 打包進 tar。**tier-5 的疑問已於 2026-07-30 解答**——在 M71 下它**不再是 no-op、而是大幅加分**（in-set heavy −0.742%、OOS +2.289% / 46-0）⇒ 候選 tar 比原先估的更好，照原鏈走（regression_suite 7 項 → `make_submission.py all` → WSL `verify_final_tar.sh` → Drive 覆蓋）。強制 `ICCAD_ADAPTIVE_CORES=48` 的那次 eval 現在**預期會與預設值不同**（會更好），不要再把「逐位相同」當通過條件。
2. **adaptive stack 常數 regen**：M71 下 in-set 品質稅 +0.967% 的來源已定位——heavy band（M42 層）**−0.742%** 由 tier-5 在 ≥40 核自動回收；mid band（tier-3）**−0.157%**，但 OOS 回收 +0.528% < 48c wall 代價 +0.695% ⇒ 放寬仍 RED（見 ledger M67-F mid 條）。⇒ **regen 的價值主要在 Final、不在 Beta**；真要做：`profile_audit.py` → `rf_score_model.py` → `m49_refine_probe.py` 三 gate，⚠️ **必須在 REFINE K=4/K=8 overlay 下做**（M67-F 更正 B）。
3. **M71 剩下四個 gated-off 旗標**（`CLUSTER_BND_CORNER` / `ANCHORED_BND_REPACK` / `CLUSTER_BND_PERMUTE` / `HPWL_SAFE_CLUSTER_SLIDE`）：⚠️ **不要照組員的做法包成 pool tier**（M72 已證 RED，見 ledger）。若要試，走 M71 的形式 = **全域 overlay**（加進 `_M71_ENV` 或作為 per-profile 變體），且**判準用 OOS 不用 local100**（M72 的教訓：in-sample 打平會藏住 1.4% OOS 差距）。
4. **低優先**：M62 micro-cap skip gate（唯一還活著的 in-window LP 形態，前置 = pre-build 時間預測器 + 機速校準）。
5. **不要做**：任何 ledger 標 RED 的軸；任何以 fp_sol 為監督的 ML；任何「縮小 LP 讓它變便宜」的變體（組員 80cc719 已證死）。

## 死路 ledger（勿重試）

> 格式：**判定** — 一行機制 — 勿重試邊界／指標。完整證據見括號內的報告與 memory。

### GREEN / 已 ship
- **🏆 M71 cluster composite-item EXPOSE+EDGE_PACK（2026-07-29 SHIPPED；`[[m71-cluster-item-expose-edgepack]]`）**——來源=組員 repo、我方獨立逐位複驗。兩個旗標（C++ 預設 OFF、binary 單獨逐位不變；wrapper `_m71_env()` 逐 profile 打開，`ICCAD_M71=0` 還原）：(a) `ICCAD_CLUSTER_BND_EXPOSE` = 排序 key 換成 `(boundary_bad, fragments, area, aspect)` + 每候選加「boundary 成員推到 item 自身對應邊」變體；(b) `ICCAD_CLUSTER_BND_EDGE_PACK` = 加「boundary 外圈／interior 中間」候選。movers 全是硬案（91/84/76/73/85/89/65），與 M63 早就定位的 case 89 純 movable cluster violator 完全吻合。in-set −1.589%、OOS −4.04%、runtime 反而更快。⚠️ 讀 OOS probe 輸出注意 `VERDICT RED` 是 M67-D 的**絕對** bar，不是 A/B 判準。
- **組員 M72 boundary-aware cluster tier ❌ RED for us（2026-07-30 我方獨立 OOS 驗證；`[[m72-tier-vs-m71-global]]`）**——組員 `b716753`+`2a0ac94`：**同一批六個 knob**，但包成 **4 隻額外 profile**（`ICCAD_M55_POOL` gate）而非 M71 的全域 overlay，動機 = 讓被 M71 弄壞的 17 案「逃生」到無 knob 的 profile（他們的 2-way per-case oracle 1.299157）。其 constructive.cpp = 我們的 + 一個 BOM ⇒ 無需重編。已移植進 repo（預設 off、off-path 七 regime × n=1..130 逐一等於 HEAD）**並修掉他們一個漏洞**：他們的 gate 只在 loop 內檢查 ⇒ `ADAPTIVE_POOL=0` 路徑會漏進那 4 隻（污染離線錨／M53 L1-L3／probe 自己的 `full` 端點）；我們在 early-return 前就讀 gate。**M67-F 同一批 80 案 held-out n>100**：pre-M71 1.659884 → **M71 1.595348** → 他們的 M72 **1.618303**（逐位重現他們回報值 ⇒ 量測本身沒問題，但**錨的是 knob-free 的 M67-G，我們早就不是了**）⇒ **M72 比我們的 M71 差 1.418%**（17 好 / 40 壞）。**tier 疊在 M71 上（m55x）= in-set n>100 20/20 逐案相等、OOS 僅 +0.057%（2 movers）、12c wall ×1.42** ⇒ 不採用（tier 留在 tree 內預設 off 當量測旋鈕）。**🔑 教訓（新 doctrine）**：**in-sample 打平可以藏住 1.4% 的 OOS 差距**——heavy band in-set M71 1.296813 vs M72 1.296769（+0.003% = 平手），OOS 卻差 1.42%。機制：全域 overlay 讓 41 隻 profile **每一隻**都帶機制（aspect×frame×knob 的組合數最大化），4 隻固定 recipe 的 tier 只有在未見案剛好合其中一隻時才有用；他們的「0 regressions」是**樣本內、由 proxy 達成非 guard 保證**。⚠️ 組員所有 headline 一律當 in-sample 看，且**要看他們錨的是哪一版**。
- **M41-M50 RF 七槍**：M41 砍 swap、M42 砍 n>100 build 冗餘、M45 band tiers（tier-3 universal + tier-4 cores≤8）、M46 C++ hot-path exact、M47 wrapper overhead、M49 REFINE band-cut（n>100 K→4）、M50 mid-band 兩層 tier。gate = **strict selection-preserving**（逐案 cost 相等才砍）。詳見 memory `[[m41-runtime-factor]]`…`[[m50-midband-refine-tiers]]`。
- **M67-F tier-5**（2026-07-26 實作 / 07-27 校準；`[[m67f-tier5-implemented]]`）：偵測 **≥40 核**才把 M42 的 22 隻放回 n>100（只動 M42 層）。**in-sample no-op**（強制 48c 亦逐位 1.326473104916827），增益 100% 在 OOS（θ_pool 0.7636），投影官方分 −0.26~−1.30%。門檻 32→40 的依據 = 實測 `c* = Σdt/max dt` 中位 19.3 / max **22.5**，≥24 核起全部 max-bound；偵測核是有效核的**上界**（本機 16 邏輯核 ≈10 有效）。🚨 **新 doctrine：高核 gate 必須 fail-CLOSED**——用 `_effective_cores_hi()`（unknown→0），**不可**沿用 `_effective_cores()`（unknown→9999，那是 tier-4「≤8」的安全方向）。gate = `m67g_tier5_gate.py`（regression_suite 第七項）。殘餘賭注 = Beta 有效並行度是否 ≥22.5。**🚨 2026-07-30 在 M71 下重量測（同 80 案 held-out n>100）：tier-5 變得嚴格更值錢**——OOS **+2.289%（46 better / 0 worse**，pre-M71 是 +2.143% / 44-1），且**不再是 in-sample no-op**：heavy band 5/20 案還原後變好 = **−0.742% of local total**（case 89 **−9.33%**、85 −4.51%、87 −3.79%、91 −2.57%、96 −0.89%）⇒ M71 正好把 M42 當年剪掉的那 22 隻「解凍」了。舊投影（官方分 −0.26~−1.30%）現在是**下界**。復現 = `m67_oos_probe.py restore --arm pool`。
- **送件 hardening**：M43（`[[m43-submission-hardening]]`）、M48 編譯鏈+binary smoke+三層安全網（`[[m48-submission-hardening]]`）、M67-A~C Linux bundled-binary-first（`[[m67c-linux-binary-green]]`）、M67-G slim 6 檔包（`[[beta-package-slim]]`）。

### 離線 GREEN、但 in-window RED（LP 家族）
- **M53 L1 品質池 ✅ offline SHIPPED**（`[[m53_l1_quality_mode]]`）：84-pool（41 base + 2 `_L1_EXTRA` + 41×K24 tier）→ 1.3176。復現 = `ICCAD_ADAPTIVE_POOL=0` + `ICCAD_L1_POOL=1` + `ICCAD_PROFILE_TIMEOUT=600`。**REFINE 向上只能走 portfolio tier**（全域 override 全池退步）；`PUSH_PASSES`/`COMPACT_ITERS` 向上逐位 no-op（early-break）。**M65**：補齊 84 池缺的兩隻 K24-extra = RED（贏處逐位 no-op、輸處全毒，期望值恰 0）。
- **M53 L2 stochastic best-of-N ❌ RED**（`[[m53_l2_stochastic_red]]`）：union oracle +0.2377% < bar 0.3%。**逐決策噪聲全域毒**（greedy 零容錯）；權重 jitter = 連續 profile 空間、8/15 重案恰 0。樣本仍作 L3 種子（GREEN）。
- **M53 L3 constraint-graph LP ✅ offline GREEN**（`[[m53_l3_lp_green]]`）：固定拓撲 + HiGHS exact 線性 HPWL → winner-only +0.76% → LP×portfolio 重選（**top-32 飽和**）1.30035 → 疊 L2 種子 **1.2978**。已探畢勿重掃：top-k>32 零貢獻、area 聯合目標僅 +0.026%、LP 2-4 pass 即 fixpoint、boundary-repair 0 可修。
- **M54 in-window LP ❌ RED**（`[[m54_lp_inwindow_red]]`）：deployable proxy-guard 形本身 GREEN（+0.891%、0 回歸）但 RF 全域淨虧；**β=0 kill test（solver 免費）最壞 cell 仍 +0.48%** ⇒ solver/C++/向量化加速軸全部 moot。復活條件：官方 median ≥~11s／RF=1.0／M62 路徑。
- **M62 break-even ⚠️ GREEN 但未動工**（`[[m62-break-even-green]]`）：M54 的「無 weak-win」只限 cap≥0.5s 網格——**cap 0.1-0.25s 的 skip gate 零加速即全 cell 弱贏**（cap=0.1 +0.06%、jitter 40/40、機速 s∈[0.5,1.5] 存活）。但 abort-only（免預測器）wins=0、s≥2 全滅 ⇒ 前置 = pre-build 時間預測器 + 機速校準。
- **組員 `80cc719` Route 1 / Route 3 ❌ 皆 RED**（2026-07-29 我方重算；`[[teammate-80cc719-route1-route3]]`）：Route 1（M71 winner + 全域 LP）1.2914 但 runtime ×2.22 ⇒ RF **+14.7~31%**；Route 3（只放開 top-12 HPWL 貢獻 block）**−0.0158%**（82% 來自 case 99 單案）+0.20s/案 ⇒ **+0.82~5.77%**。**🔑 Route 3 反向複驗 M62 的 α/β 分解**：LP 的 pairwise 建構是 **O(n²)、與幾個 unit 真的自由無關** ⇒ 縮小自由集只讓 solve 免費、build 照付，品質剩 1/50 ⇒「縮小 LP 省時」正式死。**🚨 他們的 LP run-to-run 非確定**（HiGHS `time_limit`=5.5s 在重案觸發，case 91 兩次 cost 不同）⇒ 我方未來任何 in-window LP **必須關掉 time_limit 或證明它不觸發**。唯一資產 = 離線錨 **1.2914**（不可送件）。

### quality 軸 RED（結構性）
- **M27 global packer / M40 reconstruction slicing builder**：見戰略結論 2/3（`[[m40_reconstruction_red]]`、`[[recon_index_leak_probe]]`）。
- **M57 frozen violator 重錨 LP**：41/41 infeasible。61% 是 frozen-vs-frozen 幾何**絕對死**、39% 固定拓撲 separation 鏈死。
- **M58 `compute_nsoft` 官方分母**：spec 落差真實且 live（29 案觸及、case 6 −3.31%）但 weighted **−0.0001%**（觸及全在低權重小案、n>100 恰 0）。
- **M59 REFINE rejected 種子**：251/251 是真新拓撲但 LP 後僅 +0.0055%（c1/c2 是 pre-compaction 中間態，LP 平移代替不了 compaction 拓撲改寫）。
- **M60 anchored first-pass 牆容量**：**前件空集**——全域 anchored movable violator 僅 2 個且 pack 當下在牆上 ⇒ beam/保牆排序/lookahead 全不做。⚠️ 該結論限 anchored 域（純 movable cluster 走複合 item 路徑，見 M63/M71）。
- **M61 obstacle-aware event frames**：97/97 拓撲真、96/97 開出新可行拓撲，但最佳 +0.0001%——解鎖貼牆候選的收益恆被 outline 放大的 area/HPWL 稅吃掉。⚠️ host 自身 pre-LP cost 不可當基準。
- **M63 violation 上界稽核（ABOVE-BAR / RED-leaning，未開工）**：T2 上界 −5.11% 過 bar，但 pool oracle 反證「bits 可達而恆付稅」（29/31 淨變差、全 pool realizable 僅 0.0012%）。開工前必過的 kill-test = 對 bit-clearing 候選跑 L3 LP 回收 quality 稅。**M71 已從另一側吃掉此標的的一部分**（case 89 的 4 個 cluster violator）。
- **M64 錨相鄰拓撲 flip**：529 flips **0 movers**。86.8% cone 空（鎖死者是其餘數千 pair 的 fixed-disjunct 鏈，**非** boundary 等式）、可行者全 HPWL 稅或 tie、62 個 feasible 的 vrel **全部恰 0**。單/少 pair flip 勿再試；多 pair 聯動 = M27 域。
- **M68 ML-seed 插入點（訓練前判死、零 GPU；`[[m68-ml-seed-red]]`）**：注入 fp_sol 中心 = 完美位置種子，anchor 機制**真有效**（61/100 贏 base、隔離 +2.18%）**但 vs 41-profile portfolio 僅 +0.001%、重案 0 mover** ⇒ portfolio 的 aspect/frame 多樣性早已 label-free 拿到，ML-seed 是替代非互補。
- **M52 tree-space 生成 / imitation 家族**（`[[m52_phase0_red]]`）：管線 GREEN（260 案 100% 逐位重現 fp_sol）但**容錯帶 ≈ 0**（單一 near-miss token → wR 1.232；dims 任何擾動 ≈2.9-3.1）× 輸入零訊號 ⇒ 含 direct-layout 在內全滅。⚠️ 任何 layout 級評分必用 `m52_phase0_probe._cost_strict`（`tree_decode_probe._cost_of` 不查 preplaced/fixed 硬約束）。
- **M56 per-case pool 預測**（+ **M66** 等價類補洞）：winner-only oracle 天花板巨大（@12c +3.87%）但**不可實現**——winner 身分 case-idiosyncratic，5-fold OOS 的 J\* 全部 None、breaks 30/200、最壞 +9.09%。M66 用 cost-等價類重跑仍 26/200 breaks（全是 singleton-winner LABEL-ABSENT）、kept pool ≈ chain pool ⇒ 即使 0 breaks 增益也歸零。
- **M55 drop-set OOS CV**（`[[m55_dropset_cv_oos]]`）：shipped drop set 5/5 fold 穩定，但 OOS strict-breaks 40-48% ⇒ 預言了 M67-D。**保持 `FREE_N=100`**（更高 T 嚴格更差）。
- **M67-D OOS 泛化預檢**（`[[m67d-oos-precheck]]`）：訓練集 240 案 raw 1.6533 vs in-set 1.3265 = **語料難度非過擬合**（單 profile 參照 gap 更大、portfolio 增益 OOS 更高、label floor 1.2444）。**主發現 = adaptive 切法的 OOS 品質稅 +2.825%（in-set +0.106% 的 27 倍）** ⇒「strict gate ⇒ ∀median∀cores 弱贏」**只在樣本內成立**；任何新 pool-prune 的 strict gate 結論都要自帶 ~3% OOS 折扣。
- **M67-E/F θ 實測**（`[[m67e-rf48-projection]]`、`[[m67f-theta-pool-cut]]`）：48c wall = **max-setter 100/100**；`POOL=0` +53~60%、REFINE 還原 +15~21% ⇒ **M41+M49/M50 必須留**。θ_pool **0.7636**（GREEN→tier-5）、θ_refine 0.0864（RED）、θ_mid 0.5913（品質綠但 ship RED：回收 0.620% < wall 代價 0.695%）⇒ **tier-5 只放 M42、tier-3 不動**。**2026-07-30 mid band 在 M71 下重量測：ship 判定不變（仍 RED）**——in-set drift 從 −0.0018% 漲到 **−0.157%**（8/40 案，case 68 −8.40%、79 −5.17%），但 OOS 回收 **+0.528%（21 好 / 1 壞，唯一退步 +0.05% 雜訊）< wall 代價 +0.695%** ⇒ 邊際縮小（0.76 vs 0.89）但符號沒翻。誠實範圍：+2.34% 的 mid wall 代價是 pre-M71 audit dt 推的，未在 M71 下重推。🚨 投影一律看 `restoreIdx` 變體（index 池），`restore` 是 dt-filtered 池、wall-free 是它的定義不是發現。E-core 擁塞前哨 = AMBER/lean-GREEN（max-setter 自己不變慢、slope≈0）。
- **組員 M68-M70**（`[[teammate-m68-m70-dead-ends]]`）：M68 Safe ≈ 我方 M53 L3 winner-only（獨立複驗 M54 RED）、IL68 Transformer 232 候選 0/20 勝（複驗 M52）、M69 完美結構提示上限 +0.18%（複驗 M68）、M70 selector 全部沒過 dev gate + K≤4 池 oracle ceiling 僅 77.57%（複驗 M56）。⚠️ 他們所有 headline 都在 local100（= alpha 測資 = 我方 validation set）上量 ⇒ **一律當樣本內看待直到 OOS 說話**。

### 小型死路（旋鈕/局部手法，勿重掃）
- **boundary aspect port 到舊 SA** 3.3258→3.4255（skyline ≠ shelf）；**preplaced-aligned frame**（case 89 結構性無解）；**cluster-rigid pack/slide**（cluster 無 slack + 剛體平移破壞 abutment，撞 M10 精度牆）。
- **violating boundary 修復 = 三語意封卷**：(1) 單 block 移向邊 202/202 BLOCKED；(2) LP 全域同動、block 移向邊 0 可修（限 movable violator / fixed-disjunct / non-growing bbox 語意）；(3) LP 邊移向 frozen block 41/41 infeasible。
- **env knob 軸**：WIRE_MULT 4/6、ANCHOR 0.30、ultra-narrow frame、WT/BFS/NORM/PIN 組合、CLUSTER_ORD、OM×tight 全 ≤0.063%（例外：M32 pure decoupled LR=4.5 +0.186% 已 ship）。
- **BP_WEIGHT 雙向封卷**：向上 30000→1M 無變化、向下 10000~300 standalone 0 wins（含疊 M37/fc_pin_tight 兩 stack）。
- **FRAME_ASPECTS 封卷**：standalone 全 ~0.000%；win 只在 stacked 且 aspect ≥2.0 全被 clamp 成同一 outline（**值不敏感、勿掃更多 wide 值**）。
- **free-aspect 六子軸飽和**（M33 cluster 共振 3.0 / M34 per-member 4.0 / M37 MIB wide 5.0）：per-block FREE_BOUNDARY **死**（greedy 局部 area 項與 edge-capacity 反向）⇒ boundary-aspect win 必須 **uniform**。M39 FREE_CLUSTER boundary-ungate LIVE 但 below bar（硬案零移動）已 revert。
- **FREE_RATIOS 加寬**：整池 oracle −0.044% 且抬牆 21→29.6s（free 搜尋 ∝ n²）。**proxy/_RH 已是 oracle-min**，要降分須降 oracle 本身。
- **REFINE exact 截斷 / cycle 快轉 / max_trials=3**：無 exact 截斷點（改善不飽和、fat tail）；≥2-cycle big 帶僅 2/1040、mid 帶多但要動 shipped C++；**勿在 M≥11 再疊 wall cut**（floor 飽和只付 quality 不收 RF）。
- **frame 級 B&B 早棄**：`guide.swap` 無條件執行 + 三個 nudge 破壞 bbox 單調性 ⇒ partial layout 無單調下界，可 exact 早棄的上限僅 5.8%。**勿投資 LB 證明**。
- **高核 pool-prune**（組員 codex prune V1）：CORES=12 下被砍隻無一 wall-setter ⇒ 增益恰 0.00%。⚠️ 此 RED 是 cores-conditional（低核版已由 M45 tier-4 ship）。**候選生成的 sort/TOL 比較器絕不可動**。
- **Codex M43 LTO**：單一 translation unit 上 LTO 無事可做，±1-3% 噪聲。
- **cascade/puzzle 重構提案**：與 RF 機制不符（post-proc 只佔 wall 2-7%）。

## 殘留 case（M71 前的舊值，僅供定位硬案）
89 ~1.523（preplaced boundary 撐壞 outline，最硬）、85 1.5240、65 ~1.690、62 1.5227、88 ~1.385、97 ~1.199、82 ~1.363、52 ~1.361、61 1.3134、91 1.3481。硬案 = preplaced boundary 幾何強迫；M33-M37 證實成員形狀能鬆動 89/82/88/97/79/66/65/61/52，**M71 進一步在 89/91/85/84/76/73/65 拿到分**。M57/M63 補：硬案的 violation 軸「可達但恆付稅」。

## 環境 & 指令

- **主程式**：`constructive.cpp` + `optimizer_constructive.py`（舊 SA `optimizer_claude.*` 僅 fallback）
- **Conda**：`C:\Users\Nordra\.conda\envs\iccadv\python.exe`；**Compiler**：`C:\msys64\ucrt64\bin\g++.exe`
- 組員參考碼：`C:\Users\Nordra\Downloads\teammate_iccad_study\`（其活躍 repo 在 `teammate_iccad_study\Iccadcontest2026`）
- ⚠️ eval ~2.5 分鐘（100 案 serial，M71 後 avg 1.52s）；background 用 harness `run_in_background`；`> file 2>&1` 對 native exe 印 cosmetic `NativeCommandError`（無害）

```powershell
# 編譯（Bash 工具寫 .exe 會失敗，務必用 PowerShell）
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive.exe constructive.cpp

# 官方 portfolio eval（確定性）
cd "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\iccad2026contest"
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" iccad2026_evaluate.py --evaluate ../optimizer_constructive.py 2>&1 | Select-Object -Last 12

# 快速單 profile A/B（~70 秒）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" analyze_constructive.py 2>&1 | Select-Object -First 12
```

## Gotchas
- **PowerShell 用 `;` 或 `if ($?){...}` 連接，不能用 `&&`**；Bash 工具寫 .exe 會失敗（sandbox）
- **輸出必須 `%.17g`**（非 %.10f）：否則精確 abutment 被捨入成虛假 shapely fragment（M10，單一最大 lever −6.3%）。查不明 grouping violation 先疑精度
- **proxy 選擇用 shapely vrel**（`_proxy_metrics`），不可用 C++ METRICS vrel
- **compaction 選候選用 true-cost csc** `(area+hw·hpwl)·exp(2·(bv+gf)/nsoft)`，不可用 layout_score（後者 boundary 權 150000 ≫ grouping 6500）
- 新增重 profile 前先查它自己的 per-case cpu（每案 wall ≈ 最重 profile 的 max 項，非池總量）
- **判 feasibility 用 `SolutionMetrics.is_feasible`，不可用 cost**：SA fallback 的 9.999999 是 feasible 品質上限、被 `%.4f` 進位成「10.0000」
- **任何 in-window LP 不可留 HiGHS `time_limit`**（組員 80cc719 證：觸發即 run-to-run 非確定）

## env 旋鈕

### `constructive.cpp`
- 預設：`ICCAD_BP_WEIGHT`=30000、`ICCAD_WIRE_MULT`=×1、`ICCAD_ANCHOR_W`=0.10、`ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT`=2.50/0.40
- 關後處理：`NO_COMPACT`/`NO_REFINE`/`NO_PUSH`/`NO_BND_PUSH`/`NO_SWAP`/`NO_JUMP`；`PUSH_PASSES=N`、`COMPACT_ITERS=N`、`REFINE_ITERS=N`
- pack-order：`WIRE_TIEBREAK`、`WIRE_BFS`、`BFS_PIN`、`ORDER_SWAP=K`、`ORDER_MOVE=K`、`GUIDE_MED`
- **free-aspect 六子軸**（全 gated、off=bit-identical）：`FREE_ASPECT`（single interior，`FREE_RATIOS`）／`LR_ASPECT`+`TB_ASPECT`（decoupled **uniform** boundary）／`CLUSTER_ASPECT=r`（uniform reshape 純 movable interior 成員）／`FREE_CLUSTER`+`FREE_CLUSTER_RATIOS`（per-member，layout-key 仲裁、build-time 免抬牆）／`FREE_ANCHORED`+`FREE_ANCHORED_RATIOS`+`FREE_ANCHORED_BND`（mixed cluster 成員在 wall-attach 搜 aspect）／`MIB_ASPECT=r`
- **🏆 M71 六旗標**（全部預設 0 = binary 單獨逐位不變）：`ICCAD_CLUSTER_BND_EXPOSE`、`ICCAD_CLUSTER_BND_EDGE_PACK`（wrapper `_m71_env()` 只開這兩個）；**未量過的四個** = `CLUSTER_BND_CORNER`、`ANCHORED_BND_REPACK`、`CLUSTER_BND_PERMUTE`、`HPWL_SAFE_CLUSTER_SLIDE`（見「下一步」3）。`ICCAD_M71=0` 逐位還原 pre-M71
- 死路（code 保留 gated off）：`BFS_NORM`、`CLUSTER_ORD=1/2`、`REFRAME`、`FREE_CLUSTER_BND`
- 離線探測（永不 ship）：`ORDER_FILE`+`ORDER_GLOBAL`（oracle-perm）、`ML_ANCHOR`（M68，在 `constructive_m68.cpp`）
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 退單 base profile

### `optimizer_constructive.py`（wrapper）
- **`_pool_indices()` 的 adaptive 砍法**（`ICCAD_ADAPTIVE_POOL`，預設 **1**；`=0` 還原 full 41-prof + full REFINE）：
  - M41 砍 swap（`ICCAD_ADAPTIVE_N=K` 只砍 n>K）
  - M42 砍 `_BIG_REDUNDANT_IDX`（22 隻）當 `n > ICCAD_ADAPTIVE_FREE_N`（預設 100）
  - M45 tier-3 砍 `_M45_BAND_DROP`（9 隻，60<n≤100；`ICCAD_ADAPTIVE_BAND=0` 關）
  - M45 tier-4 低核：`_effective_cores() ≤ 8` 時砍 `_M45_LOWCORE_DROP`
  - **M67-F tier-5**：`_effective_cores_hi() ≥ 40` 時**跳過** `_BIG_REDUNDANT_IDX`（`ICCAD_M67F_TIER5=0` 關）
  - `ICCAD_ADAPTIVE_CORES=N` 強制核數（tier-4/tier-5/M50 共用；`=48` 可在本機讓 tier-5 真觸發）
  - ⚠️ **`_effective_cores_hi()`（unknown→0）是高核 tier 專用，不可與 `_effective_cores()`（unknown→9999）混用**
- **`_band_env()` REFINE band-cut**：n>100 疊 `REFINE_ITERS=4`；60<n≤100 疊 `=8`（cores≤8 改 `=4`）。`ICCAD_ADAPTIVE_REFINE=0` 關
- **offline 旋鈕（永不送件）**：`ICCAD_L1_POOL=1`（84 池，須配 `ADAPTIVE_POOL=0` + `PROFILE_TIMEOUT=600`）、`ICCAD_M67F_RESTORE=1`（只跳過 M42+tier-3，量 θ 用）、**`ICCAD_M55_POOL=1`**（M72：`_M55_EXTRA` 4 隻 boundary-aware cluster profile 掛進池，idx 41-44；ship RED 見 ledger。gate 在 `ADAPTIVE_POOL=0` early-return **之前**讀，故 full-pool 路徑不會漏）
- **M48**：`ICCAD_CXX=編譯器` 插到編譯鏈最前；每次編譯成功都須過 `_binary_runs()` 1-block smoke
- ⚠️ 改 `_PROFILES` **或重編 constructive.exe** 後必須重跑 `rf_score_model.py` 重算全部 drop 常數 + `m49_refine_probe.py` 三 gate + `m67g_tier5_gate.py`

## 工具（全部永不 ship）

- **`regression_suite.py`** — 送件前一鍵七項 gate（m48 四 phase → rf_score_model asserts → m49 三 variant → m47b proxy 等價 → m67g tier-5），子行程先剝 `ICCAD_*`，~13 分鐘
- **`make_submission.py`** `stage|verify|all` — 產 `build_submission/cadc1075/`（6 檔）+ tar，verify = 官方指令 100 案逐位比對
- **`m67c_make_linux_bundle.py`** + WSL `run_all.sh` / `verify_final_tar.sh` — Linux 四關（build+smoke / m48 opwrapper / 官方 100 案 bundled-first 逐位 / 破壞 binary 落編譯鏈）
- **`m48_coldstart_dryrun.py`** — 冷啟動四 phase（含 `opwrapper` variant）
- **`rf_score_model.py`** — RF 投影 + M42/M45 drop 常數 regen + drift asserts；**`m67e_rf48.py`** — 48c 投影（`gate0/calib/fit/project/report`，投影看 `restoreIdx`）
- **`m49_refine_probe.py`** `trace|variant K [big|mid]` — REFINE band gate；**`m67g_tier5_gate.py`** — tier-5 池身分閘（V1 基準用 kill switch，不可比 HEAD）
- **`m67_oos_probe.py`** — OOS 泛化（`gate0/run/report/ref/pool0/restore`，`--pool0-lo/-hi` 選帶）；`m67_oos_cache.pkl`。**arms = `pool`（M42+tier-3 還原）/`refine`/`m55`（M72 tier + M71 全域 off = 組員原形）/`m55x`（tier 疊在 M71 上）**；無 `full` 端點時自動退成 A/B 報告並 dump `results_M72_ab_<arm>_<lo>_<hi>.json`。⚠️ cache sig 含 `repr(_PROFILES)` ⇒ **加/減 profile 會清空整個 cases 桶**；若已證 knob-off 池逐位不變，可改寫 sig 保留（2026-07-30 M72 移植即如此，備份 `m67_oos_cache.pkl.preM72`）
- **`profile_audit.py`**（audit_cache）、**`profile_vs_portfolio.py KEY=VAL`**（新 profile 增益，bar 0.05%）、`analyze_constructive.py`、`portfolio_ceiling.py`、`rh_sweep.py`、`proxy_analysis.py`（27 個工具依賴，勿刪）
- **`m53_diff_results.py`** — 兩份官方 results json 的總分/加權 delta/逐案 movers。錨：`results_L1_final.json`、`results_L3_port_top32_area.json`、`results_shipped_m51.json`、`results_shipped_preM71.json`
- **RED 存檔 probe（勿重跑求更好的數字）**：`m52_phase0_probe.py`、`m53_l2_probe.py`、`m53_l3_probe.py`、`m54_lp_rf_model.py`、`m55_dropset_cv.py`、`m56_percase_oracle.py`、`m57`~`m61`（各配 `constructive_mXX.cpp`）、`m62_break_even.py`、`m63_vio_bound.py`、`m64_flip_probe.py`、`m65_l1_cell.py`、`m66_equiv_cv.py`、`m67f_contention_probe.py`、`m68_ml_seed_probe.py`、`oracle_perm_probe.py`、`dbg_seqpair.py`、`recon_slice_probe.py`、`tree_decode_probe.py`

## 檔案結構（要點）
- `constructive.cpp` 🏆 — placer，含 M9-M37 + M46 hot-path exact + **M71 六旗標**
- `optimizer_constructive.py` 🏆 — 41-prof portfolio + shapely proxy(_RH=1.4) + `_pool_indices()` 五階 tier + `_band_env()` + `_m71_env()` + M48 三層安全網
- `constructive_m71.cpp` — 組員合併前的參照本（其餘 `constructive_mXX.cpp` 為各 probe 儀裝副本）
- `bin/constructive_linux` — Beta 包 bundled binary（`-O3 -static-libstdc++ -static-libgcc`、無 `-march`）
- `iccad2026contest/iccad2026_evaluate.py` — 官方評估腳本
- `AGENTS.md` — 給 codex 的 CLAUDE.md 副本（2026-07-22 快照，已過時、未追蹤）

## ML 現況
四種插入點全 RED（見戰略結論 6）。⚠️ 本環境**禁止跑訓練、可推論**；需要訓練時停下告知使用者上 CUDA 機。
