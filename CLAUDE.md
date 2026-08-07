# ICCAD 2026 FloorSet — Session Context

> 本檔 2026-07-29 大幅精簡。**舊版完整敘述（每條 ledger 的逐案數字、probe 全文）留在 git commit `4e2eb42` 的 CLAUDE.md**，需要考古時 `git show 4e2eb42:CLAUDE.md`。證據本體在各 `MXX_REPORT.md` 與 memory。

## Claude 對話框規範
- 聊天室語句**盡量精簡**、用**繁體中文**。

## 🚨 先讀：這題是 reconstruction，不是 floorplan optimization

- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel)`：gap=0 ∧ V_rel=0 → Cost=1.0。「找更好的解」永遠 HPWL_gap>0 → Cost>1；**還原 baseline 原圖**才能 gap≈0。
- 真天花板 **1.1079**（`fp_sol` verbatim），headroom 100% 在 quality（violation 已贏）。組員 1.0322 是 **label oracle**（hidden test 不適用）；legit 上限 ~1.62。
- 但 **reconstruction 本身 RED**（M40）：X 結構無法從 connectivity 還原、Y 序需 label。⇒ 走「更好的 placer / 更聰明的 portfolio」而非「還原」。

### 現況一句話（2026-08-06，M80 收工 + R=512 續挖判 RED 之後）
- **Beta 已結案**：上傳的是 1.3054（M71/M73，md5 `ba694bc6…`）。**使用者 2026-08-01 裁示不換件** ⇒ 那顆就是 Beta 的最終答案，不要再動它。
- 🏆 **M80 已進 tree（GREEN）**：M79 撞到的「旋鈕空間隨機聯合抽樣」做成 **cores-gated tier**
  （8 隻 profile，`_M80_IDX` = idx 86-93，`_effective_cores_hi() >= 40` 才開）。
  **OOS 240 案 ×2 份 disjoint 樣本 @48c：NET +1.786%（s1）／+1.909%（s2）**，bar 0.30% ⇒ 過 6 倍。
  **這是自 M71 以來最大的單一 lever。**
- **local 分數要分兩個數字講**：低核（本機 16、WSL）**逐位不變 1.293461035226291**（tier 惰性）；
  **48 核形狀 = 1.266623425**（−2.075%）。評分機是 48 核 ⇒ 真正出貨的是後者。
- **換件包尚未包含 M80**：`build_submission/` 那顆還是 M74（`op_wrapper.py` md5 `ce4f3471…`）。
  Final（8/21）要送的話**必須重打包**，見「下一步 1」。
- **ML 全線 RED，M80 把它釘得更死**：cloud 從 R=128 → 256 → **512**，per-case oracle 一路
  +2.03% → +2.649% → **+3.081%**，但 LOO 預測器三次都文風不動（global 恆 +0.166%／
  band 0.051→0.044%／knn5 0.127→0.128%）⇒ **oracle 與可預測值的差距三次都是變寬**。組員也已用 oracle 天花板自判 ML-as-placer 死。
  `m77_*` 兩支判定工具維持待命。
- 四大軸狀態：**quality 軸**——M26/M27/M40 三面天花板仍成立；M71 軸由 M75 關閉、
  M76 關掉 escape tier、M78 關掉 anchored 候選集合；**「profile 聯合空間」這個洞由 M79 打開、M80 兌現，
  但「加大 R」這條已於 08-06 探畢＝RED（R=512 held-out 反而變差）；還活著的只有「換提案分布」**；
  **RF 軸**——M41-M50 七槍 + tier-5 已 ship；**LP 軸**——offline GREEN（錨 1.2914）、in-window RED（M54/M62/組員 80cc719）；**ML 軸**——**五種**插入點全 RED（M52 生成、M56 selector、M68 seed、LP refinement、**M79 hyper-heuristic**）。
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

### 🏆 **48 核 1.266623425**（M80 knob-cloud tier，2026-08-05，**未出貨**）
- 全文 `M80_REPORT.md`。8 隻隨機聯合抽樣的 profile 掛成高核 tier。**低核逐位不變**
  （預設 16c 與強制 48c+tier OFF 都是 `1.293461035226291`、0 movers）；
  **48c + tier ON = `1.2666234251`**（−2.075%，56 好 / 2 壞、100/100 feasible），
  與離線 greedy 曲線 K=8 **逐位相同**。
- **OOS 240 ×2 樣本 @48c**：s1 `1.555854672`→ quality +2.073%、dRF +0.287%、**NET +1.786%**；
  s2 `1.557813659`→ +1.920% / +0.011% / **NET +1.909%**。三個帶都是正的（重帶 +2.0% / +1.8%）。
- **必須 cores-gated**：同一批向量 @12c 是 dRF **+10.619%**、100/100 案被抬 wall ⇒ **NET −8.544%**。
  gate 用 `_effective_cores_hi()`（unknown→0，fail-CLOSED），與 tier-5 共用 ≥40 核這個賭注。
- **零 cache 作廢**：append 在 `_M55_BASE_LEN` 之後 ⇒ 四顆離線 cache 全部仍有效（見 ledger）。

### 🏆 local **1.3054**（M71，2026-07-29，commit `0ff45f4`）
- 機制 = `make_group_item()` 純 movable cluster 複合 item 的**候選集合 + 排序 key**（見 ledger「M71」）。
- 驗證：旗標關 → `1.326473104916827` 0 movers；開 → `1.305389893450635`（與組員 json 逐位相同）、100/100 feasible、avg 1.75→**1.52s**（品質與 RF 雙贏）；OOS 240 案 `1.653329→1.586461`（**−4.04%**，是 in-set 的 2.5 倍）。
- ~~⚠️ 已量化缺口：adaptive stack 的 in-set 品質稅 +0.967%~~ ⇒ **2026-07-30 由 M74 regen 收掉，剩 +0.046%**（1.293461 vs `ADAPTIVE_POOL=0` 的 1.2929）。

### 🏆 local **1.293461**（M74 adaptive 常數 regen，2026-07-30，**未出貨**）
- 全文 `M74_REPORT.md`。預設（≤16 核）**1.293461**（vs M73 的 1.305390 = **−0.769%**，avg 1.52→1.45s）、`ADAPTIVE_CORES=48` 也是 **1.293461**（vs 1.295548 = −0.158%）、`ADAPTIVE_POOL=0` 天花板 1.2929 ⇒ **品質稅 +0.967% → +0.046%**。逐案 **14 movers 全部變好、0 退步**，100/100 feasible，regression_suite **7/7**。
- 改了什麼：`_BIG_REDUNDANT_IDX` 成員大換（仍 22 隻）、tier-4 三帶重推、tier-3 內容 9→15 **且降級為 cores-gated**（新常數 `_M45_MID_CORES_MAX=16`）、mid REFINE **K=8→6**。
- **🔑 最大教訓**：**strict in-sample gate 完全不保證 OOS**。tier-3 新的 15 隻剪法 in-set 嚴格等價，但同 80 案 held-out mid 帶「跑滿 pool」反而好 **−0.702%（30 好 / 0 壞）**；而它在 48 核只把 wall 從 1.32 買到 1.30s（c\* max 15.2 ⇒ max-setter-bound，投影 +0.00%）⇒ 純付品質 ⇒ 降級成低核專用。**這是 tier-5 邏輯的鏡像**。
- OOS 240 案：**48c 評分機形狀 ALL −2.068%（71 好 / 5 壞）**、mid −0.583%。⚠️ 其中 n>100 的 −2.238% 是對「舊常數 @16 核」比的；真 48 核上舊常數也會由 tier-5 還原同一組 35 隻 pool ⇒ **重帶在評分機上新舊相同**，48c 的真增益 = mid 的 −0.583%。
- 副產物（長期價值）：cache 簽章現在釘 **exe md5 + overlay 常數**。舊簽章只有 `repr(_PROFILES)`，所以 07-10 的 cache 對上 07-29 的 M71 exe 完全偵測不到，**所有離線 gate 一直在量 pre-M71 的 placer**（`profile_audit` / `m49_refine_probe` 都漏了 `_m71_env()`）。

### 📦 送件狀態（Beta 已上傳；**Final deadline 2026-08-21**）

> **🚧 2026-08-01 進行中：M74 換件包已在本機備妥並驗過，尚未上傳。**
> 交付物 `build_submission/cadc1075.tar.gz`（305709 B、6 檔），**`op_wrapper.py` md5 `ce4f34716ea14863e62f68d6970e983d`**（tar md5 `d529c7828e2e8a36a7165e70d9a22ee0`，不可重現、僅供本輪比對）。
> 已過：`regression_suite.py` **7/7**、`make_submission.py all` + 再一次 `verify`（官方指令逐位 `1.293461035226291`、100/100 feasible）、`m67c_make_linux_bundle.py`（bundle `C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz`，md5 `b3989af0a0958488a77df6629cef6d04`，**內嵌 tar 與磁碟上那顆同一個 md5**）。
> **還沒做**：GPU 機 WSL 的 `verify_final_tar.sh`（含 `final48`）、Drive 覆蓋。
> 所有錨已改指 M74：`make_submission._ANCHOR`、`m67c` 的 `ANCHOR`/`ANCHOR48`/`_SOURCES`。

### 📦 已上傳的舊包（Beta，2026-07-30）
- **✅ 已上傳 = M73 包（M71 + tier-5），2026-07-30 覆蓋成功**：6 檔、tar md5 `ba694bc6c4c40485b12146d6696dbf7b`（299257 B）、**`op_wrapper.py` md5 `c2e27c9993afd20b5c14934f6ceea8c3`**。⚠️ **tar md5 不可重現**（gzip 內嵌 mtime，每次 stage 都變）⇒ **身分一律看 op_wrapper md5**。
- 驗證矩陣（Windows / GPU 機 WSL2 **雙邊逐位相同**）：預設 `1.305389893450635`（\|d\|=0、0 ULP warn）、`ICCAD_ADAPTIVE_CORES=48`（tier-5 觸發）`1.295547821428148`；兩輪皆 feasible 100/100、bundled-first OK（無 `constructive.exe` 產物）。48c 的 movers 恰 5 案 85/87/89/91/96、n≤100 逐位不變。
- 換件鏈（已跑完一輪，Final 照走）= `regression_suite.py`（7 項）→ `make_submission.py all` → `m67c_make_linux_bundle.py` + GPU 機 WSL `verify_final_tar.sh`（現含 round 2b = `final48`）→ Drive 覆蓋。
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

里程碑一行：M1 3.62 → M10 `%.17g`+compaction 1.4528 → M13 proxy oracle 1.4349 → M24 HPWL jump 1.3862 → M29-M37 free-aspect 六子軸 1.3269 → M41-M50 RF 七槍（local 1.3285 = RF fiction、avg 9.89→1.49s）→ M51 wide-CLAMP 1.3265 → **M71 cluster-item 1.3054** → **M74 adaptive 常數 regen 1.2935**（未出貨）→ M75 M71 殘餘四旗標全 RED（軸關閉，分數不動）→ M76 組員 escape tier RED（48 核形狀只剩 +0.10%，被 tier-5 吃掉）→ M78 候選集合第二條路徑 RED（唯一贏的 `anch_cross` OOS 只有 −0.160%，且「加候選」預設有害）→ **M79 自建 ML Gate 0：形狀 oracle +0.099% RED、逐案旋鈕 oracle +2.03% 但不可預測 RED，副產物「隨機聯合抽樣的固定 profile」held-out NET +0.655% ⇒ 古典線重開** → **🏆 M80 把它做成 cores-gated tier：48 核 1.2666（−2.075%）、OOS 240×2 樣本 NET +1.79~1.91%，自 M71 以來最大 lever**。

## 🔑 戰略結論（哪些軸封了、哪些沒）

1. **ordering / ML ranking 封卷**（M26 oracle-perm）：注入完美 fp_sol 排序只多拿 +0.005% ⇒ 瓶頸是 placer 不是 order。誠實範圍：只測兩個 scalar sort key；anchored first-pass 內部順序的洞已由 M60 補診（前件空集）。
2. **packer 重寫封死**（M27）：greedy 已在 (area,HPWL) frontier、agap 與 hgap 結構耦合 ⇒ B*-tree/SP/skyline 不值得。誠實範圍：`dbg_seqpair.py` 是近似語意；M59/M61/M64 以官方 strict eval 攻鄰近軸亦全 RED。
3. **reconstruction RED**（M40）：X 從 connectivity 不可還原（Spearman 0.009）+ Y 序需 label（+159%）。
4. **RF 軸 = 已兌現的主力**（M41-M50 + M67-F tier-5）：alpha 實測 RF 0.7081≈floor 證明七槍全數兌現。
5. **🚨 但 quality 軸沒有 converged**（M40 的「converged」已被 M71 推翻）：M33-M39 掃的全是**成員 aspect**，沒人動過複合 item 的**候選集合與排序 key**。M71 −1.589% 就在那個洞裡。⇒ 找洞要找「從沒被參數化的結構決策」，不是再掃已知旋鈕的值。
6. **ML 五種插入點全 RED**：生成（M52 imitation 零容錯×零訊號）、selector（M56 winner case-idiosyncratic + proxy 已 oracle）、seed（M68 完美種子 vs portfolio 僅 +0.001%）、refinement（LP 系列 + M64）、**hyper-heuristic（M79：逐案預測 profile，oracle +2.03% 但 LOO 預測器 +0.09~0.17%，輸給不看案子的固定 profile ⇒ M56 的完整重演）**。重開唯一條件 = 取得 rival 1.29 的 legit 方法細節。
7. **🚨 M79 新增：「餵給 packer 的決策」整層已經吃乾**——三個 perfect-information 上界
   排在一起就是結論：完美排序 +0.005%（M26）／完美位置種子 +0.001%（M68）／
   **完美形狀 +0.099%（M79）**，而到 label floor 的 headroom 是 **14.343%**。
   ⇒ 缺的不是逐 block 的參數，是**版圖拓撲本身**（M27 的另一面）。任何「ML 預測某個
   per-block 量再交給我們的 placer」的提案，天花板都在 0.1% 量級，**不必再量**。

## 下一步（依 ROI）— 更新於 2026-08-06

> **只列還沒做的事。** 已收斂的軸（M74 / M75 / M76 / M78 / M79 的兩個 ML 分支、
> **M80 的「加大 R」續挖（R=512 RED，2026-08-06）**）在死路 ledger，不在這裡。
> **本節在每個 milestone 收尾時必須改寫**，見 `[[keep-next-steps-current]]`。

1. **Final 送件包：現在必須重打包（M80 進來之前的包已經過期）**
   `build_submission/` 那顆是 **M74**（`op_wrapper.py` md5 `ce4f3471…`），**不含 M80**。
   評分機是 48 核 ⇒ 送舊包等於白白丟掉 **−2.075%**。
   **要做的事**：`regression_suite.py`（**八**項）→ `make_submission.py all` →
   `m67c_make_linux_bundle.py` → GPU 機 WSL `verify_final_tar.sh` → Drive 覆蓋。
   ⚠️ **錨要一起換**：`make_submission._ANCHOR`、`m67c` 的 `ANCHOR`/`ANCHOR48`/`_SOURCES`
   目前都指 M74；**`final48` 那輪的預期值會從 `1.295547821428148` 變成 M80 的 48 核值**
   （tier 開著 ⇒ 不再是 M74 的 1.293461）。
   ⚠️ **只差 GPU 機 WSL 的 `verify_final_tar.sh`** —— 這台**沒有 WSL / Docker / Linux bash**
   （08-03 實測 `wsl -l -v` 無 distro），做不了。
   👉 runbook：`FINAL_LINUX_VERIFY_RUNBOOK.md`（**內含的預期逐位值要跟著 M80 更新**）。
   ⚠️ **Beta 已過、使用者裁示不換件** ⇒ 這包是給 **Final（8/21）** 的，不要去覆蓋 Beta。

2. **M62 micro-cap skip gate（唯一還活著的技術項目，但先過雙標關）**
   前置 = pre-build 時間預測器 + 機速校準。⚠️ 離線增益只有 **+0.06%**，而我們已經用
   同一把尺殺掉 M76 的 **+0.087%** 與 M78 的 **+0.160%（OOS）**
   ⇒ **要開工就得先寫清楚為什麼標準不同**（例如：M62 是 median-independent 的 RF 增益，
   不是 OOS 品質增益，兩者的 bar 本來就不該相同——但這個論證必須先成立才動手）。

3. **Beta 成績（外部，很可能等不到）**
   使用者 08-03 回報：組員認為 Final 前可能拿不到。若真的拿到，就用官方分反推真實
   median 與 RF，重新校準 `m67e_rf48.py`（現在錨的是 M10 廉價池那版的逐案 runtime，
   p50 0.673s，不是現行 shipped），才知道 tier-5 的賭注（有效並行度 ≥22.5）成不成立。

4. **ML 候選判定 — 已停放，不是待辦（M79 之後更沒有理由重啟）**
   組員 08-01（`b86a02c`）用 oracle 天花板自己判死 ML-as-placer。M79 又把**我方自己
   做一顆**的兩條路都關了（形狀上界 +0.099%、旋鈕 oracle 上界 +2.03% 但逐案不可預測）。
   `m77_ml_candidate_probe.py`（in-set）與 `m77_oos_probe.py`（OOS 240 兩份樣本）機制都
   驗過、隨時可用，**但沒有候選會進來**。若日後拿到學長 `cadc1106` 的產物才重啟；
   屆時判定看 **s2**（worker_10..19，對他們的模型才是真 OOS）。

5. **M80 打開的其他抽樣方向（有想法再做）**
   🚨 **先讀 ledger 的 R=512 條目：「加大 R」這條已經探畢且是 RED**——in-sample 還在漲
   但 5-fold held-out 在 K=7~12 全面**變差**（K=8 1.000%→0.934%、K=12 1.293%→1.015%），
   機制是 fold 內貪婪在更大的候選集上過擬合。⇒ **要動就動提案分布，不是樣本數。**
   目前的提案分布是**一半擾動出貨 profile 的 1-3 個 knob、一半從 per-knob 先驗重抽**，
   而且**排除了 `ORDER_SWAP`/`ORDER_MOVE`**（5-12s/案，會自己當 48 核 max-setter）。
   還沒試過的：改提案分布（例如以 M80 挑中的 8 隻為新的擾動中心再抽一輪）、
   把 `_NUMERIC` 的夾取範圍放寬、允許 cloud 帶 C++ 旗標（M75 判死的四個是**單獨**死）。
   ⚠️ 每一條都要走同一把尺：OOS 240 ×2 樣本 @48c、NET bar 0.30%、K 看手肘；
   而且**判定要看 5-fold held-out，不是 in-sample greedy 曲線**（R=512 就是這樣現形的）。

6. **不要做**：任何 ledger 標 RED 的軸；任何以 fp_sol 為監督的 ML（**使用者 08-05 裁示：
   完全禁止，訓練訊號只能 self-supervised**；離線 oracle 探測用 label 不受限）；
   任何「縮小 LP 讓它變便宜」的變體（組員 80cc719 已證死）；
   **在這台機器上跑大模型訓練**（小模型 CPU 訓可以，使用者 08-05 裁示）。
   **M78 三條**：(a) 再加候選位置（`item_candidates` 兩種加法都是 +0.36~0.40%，
   出貨候選集合是調過的不是貧乏的）；(b) anchored 成員排序 key（整條封卷，
   對照組拿掉 boundary 優先序是 +1.069%）；(c) 把 `anch_cross` 包成 pool tier
   （oracle 上界 +0.384% 但要 41 隻 twin，max-setter 的 twin 一進池就 ΔRF +2.04%）。
   **M79 兩條**：(d) ML 預測 per-block 形狀／排序／位置種子（三個 perfect-information
   上界分別是 +0.099% / +0.005% / +0.001%）；(e) ML 逐案預測 profile
   （oracle +2.03% 但 LOO 預測器只有 +0.09~0.17%，**輸給不看案子的固定 profile**）。

> 🗓️ **Final deadline = 2026-08-21**（組員 `HANDOFF_2026-07-30.md`，我方原本只記了 Beta）。
> 📋 **每個 session 開始前**：`cd C:\Users\Nordra\Downloads\teammate_iccad_study\Iccadcontest2026 && git fetch origin`，再讀 `git show origin/main:ATTEMPTED_ROUTES_AND_DEAD_ENDS.md`（本地 clone 常落後數個 commit；該檔本身也可能停在舊日期，要另看新 commit 的 README）。

## 死路 ledger（勿重試）

> 格式：**判定** — 一行機制 — 勿重試邊界／指標。完整證據見括號內的報告與 memory。

### GREEN / 已 ship

- **🏆 M80 knob-cloud cores-gated tier（2026-08-05 GREEN，已進 tree、未出貨；`M80_REPORT.md`、`[[m80-knob-cloud-tier]]`）**——把 M79-B′ 兌現成 8 隻固定 profile 的高核 tier（`_M80_EXTRA`/`_M80_IDX`=idx 86-93/`_M80_CORES_MIN`=40/`_m80_active()`，`ICCAD_M80_TIER` kill switch、`ICCAD_M80_MIN_N` 帶別 gate）。**先把 cloud 從 R=128 加到 256**（`build_cloud` 在 R 上 prefix-stable，只付新向量的錢：12800 runs / 797s），K=8 in-sample 從 +1.576% 漲到 **+2.075%**、5-fold held-out 從 0.791% 漲到 **1.000%**。**OOS 240 案 ×2 份 disjoint 樣本 @48c：s1 quality +2.073% / dRF +0.287% / NET +1.786%；s2 +1.920% / +0.011% / NET +1.909%**（bar 0.30% ⇒ 過 6 倍，三個帶全正）。**K=8 是在 OOS 上挑的**：兩份樣本都在 K=8 有乾淨手肘（第 8 隻值 +0.195pp/+0.249pp，第 9 隻只值 +0.004pp/+0.009pp），K=12 只多 +0.019pp 卻讓池大 50%。**必須 cores-gated**：同一批向量 @12c dRF **+10.619%**、100/100 案被抬 wall ⇒ NET **−8.544%**（獨立重現 M79 手推的 +10.614%）。驗證：官方 eval 三輪（預設 16c 與強制 48c+tier OFF 都逐位 `1.293461035226291` 0 movers；48c+tier ON `1.2666234251` 與離線 greedy K=8 **逐位相同**，56 好/2 壞、100/100 feasible）、`m80_tier_gate.py` V1-V6 ALL PASS、`m80_tier_probe selftest` K=0 逐位重現 m77 的 `1.555854672`。**🔑 教訓 A：「單獨死不代表聯合死」**——被挑中的 `#100` 同時帶 `BP_WEIGHT=274048`、`MIB_ASPECT` tall 側 0.2338、frame scale 1.45，**這三條在 ledger 裡各自都被判過死**；凡是「某旋鈕封卷」的結論**只對單軸掃描成立**。**🔑 教訓 B：零 cache 作廢的落地方式**——append 在 `_M55_BASE_LEN` **之後**，四顆離線 cache（`audit_cache{,_ship,_esc}` / `m67_oos_cache{,_c48}` / `m77_oos_audit` / `m79_knob_cloud`）簽章全錨出貨前綴 ⇒ **一顆都不失效**，省掉 30-35 分鐘的 regen 鏈；而且進前綴零好處（48 核上唯一還活著的前綴剪法是 M41 的 content-based swap 過濾，cloud 本來就排除 ORDER_SWAP/MOVE ⇒ 與 drop-常數推導鏈零交互）。**🔑 教訓 C：M78 的「加候選預設有害」不可外推到 portfolio 層**——M78 講的是 packer **內部候選位置**（greedy 短視），M80 加的是**整隻 profile**，而 proxy 仲裁在異質候選上是 oracle-perfect（M76/M77 驗過）⇒ 這層是弱單調的。**但 `hmin` 耦合仍真實**（proxy 的 `hmin` 是整池 min HPWL，新候選壓低它會等比放大所有候選的 hpwl 項卻不動 area 項 ⇒ 既有候選排序可翻），實測 in-set 2 案、OOS 各 2-5 案變差 ⇒ `m67_oos_probe` 的 M80 arm **刻意不放進 strict「永不變差」分支**。⚠️ 與 tier-5 共用「評分機有效並行度 ≥40」這個賭注；賭輸則兩者一起不觸發，增益歸零但**不會變負**。
  **🚨 2026-08-06 續挖 R=256 → 512 = RED，`_M80_EXTRA` 維持 K=8 / R=256 不變**（`m80_cloud512_run.txt` / `m80_greedy512.txt` / `m80_loo512.txt`，向量順序另存 `m80_vectors_R512.json`，`m80_vectors.json` 已還原、gate ALL PASS）。25600 筆新 runs / 1465s，prefix-stability 逐位成立（incumbent 仍 `1.293461035`、榜首仍是 +0.439% 的 `#100`、greedy 前 5 隻順序不變、第 6 隻才換成新抽到的 `#307`）。**in-sample 幾乎不動：K=8 +2.075% → +2.086%（+0.011pp）**、K=12 +2.368% → +2.423%、K=16 +2.677%。**但 5-fold held-out 在真正要用的 K 全面變差：K=8 1.000% → 0.934%、K=9~12 0.978/0.988/0.988/1.015%（R=256 同 K 是 1.000/1.117/1.304/1.293%）**，只有 K=4~6 略好。⇒ **候選集加倍讓 fold 內貪婪過擬合**：同一批 80 案上它拿得到更高的訓練增益（因為 R=256 的向量是 R=512 的子集，訓練分數單調不減），轉移下來反而更少。**🔑 教訓：in-sample greedy 曲線在這個實驗裡是失效指標**——只看它會判 R=512 微贏，看 held-out 才知道是輸；M80 當初「K 在 OOS 上挑、看手肘」的紀律，在**樣本數**這個維度上同樣適用。**副作用是把 ML 那半邊釘得更死**：per-case oracle 從 +2.649% 再漲到 **+3.081%**（1.253613518，81/512 distinct winners），三個 LOO 預測器卻文風不動（global **+0.166%**、band +0.051%→+0.044%、knn5 +0.127%→+0.128%）⇒ **R 從 128→256→512 三次加倍，oracle 與可預測值的差距三次都是變寬**。cloud cache 現在是 512 隻 / 51200 runs（145 MB），未來要換提案分布可直接沿用。
- **🏆 M79-B′ 旋鈕空間的隨機聯合抽樣（2026-08-05 GREEN，**已由 M80 兌現並出貨進 tree**；`M79_REPORT.md`、`[[m79-shape-and-knob-ceilings]]`）**——M79 ML 探測的副產物，**是古典增益不是 ML**。把 R=128 個隨機聯合抽樣的旋鈕向量（一半是出貨 profile 的 1-3 knob 擾動、一半從 per-knob 先驗重抽；**排除 ORDER_SWAP/MOVE**，5-12s/案會自己當 48 核 max-setter）用**固定 profile** 的形式貪婪加進池：in-sample K=1/4/8 = **+0.439% / +1.106% / +1.576%**，**5-fold CV held-out = +0.234% / +0.459% / +0.791%**（轉移率 **50%**，5 個 fold 挑到的向量高度重疊 ⇒ 不是雜訊），dRF@48c = +0.039% / +0.050% / +0.136% ⇒ **held-out NET@48c = +0.195% / +0.409% / +0.655%**，K≥4 過 OOS ship bar。**單一最佳新向量 in-sample +0.439% = M30/M31 掃到飽和時最好那隻（≤0.063%）的 7×**。**🔑 為什麼會漏掉 30 個 milestone**：M30/M31 是**逐 knob 從人工堆疊的 recipe 往外掃、低於 0.05% 就停**，隨機**聯合**抽樣會走到座標式貪婪永遠不會造訪的組合——挑中的 `#100` 同時把 `BP_WEIGHT` 拉到 274048、`MIB_ASPECT` 往 tall 側 0.2338、frame scale 放寬到 1.45，**這三條各自都在 ledger 裡被判過死**。⇒ **「單獨死不代表聯合死」，凡是「某某旋鈕封卷」的結論都只對單軸掃描成立。** ⚠️ **12 核上這條是大負**（K=8 dRF **+10.614%**、100/100 案被抬 wall）⇒ ship 形態必須是 cores-gated tier，與 tier-5 共用同一個賭注。⚠️ +0.791% 是**同語料 5-fold**，真 OOS 240 還沒跑（`--force-cores 48` 必要）。下一步見「下一步 0（M80）」。
- **🏆 M74 adaptive 常數 regen（2026-07-30 GREEN，未出貨；`M74_REPORT.md`、`[[m74-adaptive-regen]]`）**——全部 drop 常數在 **M71 + 出貨 REFINE overlay** 下重推（新增 `audit_cache_ship.pkl`；`profile_audit.py base|ship`）。local 1.305390 → **1.293461**（−0.769%，14 movers 全好 0 壞，avg 1.45s），品質稅 +0.967% → **+0.046%**，7/7 gate。改動：`_BIG_REDUNDANT_IDX` 成員大換、tier-4 三帶重推、tier-3 9→15 **且降級 cores-gated（`_M45_MID_CORES_MAX=16`）**、mid REFINE K=8→**6**（big K=4 在 M71 下變成**純贏** −0.056%，符號翻了）。**🔑 新 doctrine（M72 的加強版）：strict in-sample 等價 ≠ OOS 等價**——tier-3 新剪法 in-set 嚴格相等，OOS mid 帶卻輸給滿 pool **−0.702%（30 好 0 壞）**，而它在 48 核只買到 1.32→1.30s（c\* max 15.2，投影 +0.00%）⇒ 高核純付品質 ⇒ 降級（tier-5 邏輯的鏡像）。**🚨 附帶發現：在此之前所有離線 gate 都在量 pre-M71 的 placer**——`profile_audit.run_one()` / `m49_refine_probe.run_case()` 都漏了 `_m71_env()`，而 cache 簽章只有 `repr(_PROFILES)`（無 exe md5、無 overlay）所以偵測不到；簽章已修成釘 exe md5 + overlay 常數，`m67_oos_probe._sig()` 也補上全部 adaptive 常數。⚠️ **`regression_suite.py` 一定要用 PowerShell 跑**：Bash 工具的 sandbox 擋 `.exe` 寫入 ⇒ m48 必假 FAIL。
- **🏆 M71 cluster composite-item EXPOSE+EDGE_PACK（2026-07-29 SHIPPED；`[[m71-cluster-item-expose-edgepack]]`）**——來源=組員 repo、我方獨立逐位複驗。兩個旗標（C++ 預設 OFF、binary 單獨逐位不變；wrapper `_m71_env()` 逐 profile 打開，`ICCAD_M71=0` 還原）：(a) `ICCAD_CLUSTER_BND_EXPOSE` = 排序 key 換成 `(boundary_bad, fragments, area, aspect)` + 每候選加「boundary 成員推到 item 自身對應邊」變體；(b) `ICCAD_CLUSTER_BND_EDGE_PACK` = 加「boundary 外圈／interior 中間」候選。movers 全是硬案（91/84/76/73/85/89/65），與 M63 早就定位的 case 89 純 movable cluster violator 完全吻合。in-set −1.589%、OOS −4.04%、runtime 反而更快。⚠️ 讀 OOS probe 輸出注意 `VERDICT RED` 是 M67-D 的**絕對** bar，不是 A/B 判準。
- **組員 M72 boundary-aware cluster tier ❌ RED for us（2026-07-30 我方獨立 OOS 驗證；`[[m72-tier-vs-m71-global]]`）**——組員 `b716753`+`2a0ac94`：**同一批六個 knob**，但包成 **4 隻額外 profile**（`ICCAD_M55_POOL` gate）而非 M71 的全域 overlay，動機 = 讓被 M71 弄壞的 17 案「逃生」到無 knob 的 profile（他們的 2-way per-case oracle 1.299157）。其 constructive.cpp = 我們的 + 一個 BOM ⇒ 無需重編。已移植進 repo（預設 off、off-path 七 regime × n=1..130 逐一等於 HEAD）**並修掉他們一個漏洞**：他們的 gate 只在 loop 內檢查 ⇒ `ADAPTIVE_POOL=0` 路徑會漏進那 4 隻（污染離線錨／M53 L1-L3／probe 自己的 `full` 端點）；我們在 early-return 前就讀 gate。**M67-F 同一批 80 案 held-out n>100**：pre-M71 1.659884 → **M71 1.595348** → 他們的 M72 **1.618303**（逐位重現他們回報值 ⇒ 量測本身沒問題，但**錨的是 knob-free 的 M67-G，我們早就不是了**）⇒ **M72 比我們的 M71 差 1.418%**（17 好 / 40 壞）。**tier 疊在 M71 上（m55x）= in-set n>100 20/20 逐案相等、OOS 僅 +0.057%（2 movers）、12c wall ×1.42** ⇒ 不採用（tier 留在 tree 內預設 off 當量測旋鈕）。**🔑 教訓（新 doctrine）**：**in-sample 打平可以藏住 1.4% 的 OOS 差距**——heavy band in-set M71 1.296813 vs M72 1.296769（+0.003% = 平手），OOS 卻差 1.42%。機制：全域 overlay 讓 41 隻 profile **每一隻**都帶機制（aspect×frame×knob 的組合數最大化），4 隻固定 recipe 的 tier 只有在未見案剛好合其中一隻時才有用；他們的「0 regressions」是**樣本內、由 proxy 達成非 guard 保證**。⚠️ 組員所有 headline 一律當 in-sample 看，且**要看他們錨的是哪一版**。
- **M41-M50 RF 七槍**：M41 砍 swap、M42 砍 n>100 build 冗餘、M45 band tiers（tier-3 universal + tier-4 cores≤8）、M46 C++ hot-path exact、M47 wrapper overhead、M49 REFINE band-cut（n>100 K→4）、M50 mid-band 兩層 tier。gate = **strict selection-preserving**（逐案 cost 相等才砍）。詳見 memory `[[m41-runtime-factor]]`…`[[m50-midband-refine-tiers]]`。
- **M67-F tier-5**（2026-07-26 實作 / 07-27 校準；`[[m67f-tier5-implemented]]`）：偵測 **≥40 核**才把 M42 的 22 隻放回 n>100（只動 M42 層）。**in-sample no-op**（強制 48c 亦逐位 1.326473104916827），增益 100% 在 OOS（θ_pool 0.7636），投影官方分 −0.26~−1.30%。門檻 32→40 的依據 = 實測 `c* = Σdt/max dt` 中位 19.3 / max **22.5**，≥24 核起全部 max-bound；偵測核是有效核的**上界**（本機 16 邏輯核 ≈10 有效）。🚨 **新 doctrine：高核 gate 必須 fail-CLOSED**——用 `_effective_cores_hi()`（unknown→0），**不可**沿用 `_effective_cores()`（unknown→9999，那是 tier-4「≤8」的安全方向）。gate = `m67g_tier5_gate.py`（regression_suite 第七項）。殘餘賭注 = Beta 有效並行度是否 ≥22.5。**🚨 2026-07-30 在 M71 下重量測（同 80 案 held-out n>100）：tier-5 變得嚴格更值錢**——OOS **+2.289%（46 better / 0 worse**，pre-M71 是 +2.143% / 44-1），且**不再是 in-sample no-op**：heavy band 5/20 案還原後變好 = **−0.742% of local total**（case 89 **−9.33%**、85 −4.51%、87 −3.79%、91 −2.57%、96 −0.89%）⇒ M71 正好把 M42 當年剪掉的那 22 隻「解凍」了。舊投影（官方分 −0.26~−1.30%）現在是**下界**。復現 = `m67_oos_probe.py restore --arm pool`。
- **送件 hardening**：M43（`[[m43-submission-hardening]]`）、M48 編譯鏈+binary smoke+三層安全網（`[[m48-submission-hardening]]`）、M67-A~C Linux bundled-binary-first（`[[m67c-linux-binary-green]]`）、M67-G slim 6 檔包 → **M73 換件（`[[beta-package-slim]]`）**。
- **M73 換件（2026-07-30 SHIPPED；`[[audit-cache-sig-shipped-pool]]`）**——把 M71+tier-5 打包上傳。路上修掉三個**只有真的跑換件鏈才會現形**的問題：(a) **`audit_cache.pkl` 的 sig 用 `repr(_PROFILES)`**，M72 append 4 隻 gated-off profile（41→45）就讓 **rf + m49×3 共 4/7 gate FAIL**（`cache profile signature != current pool`）⇒ 六個離線 gate 全改錨出貨池 `_PROFILES[:_M55_BASE_LEN]`，守衛語意不變、**不需重跑 `profile_audit.py`**（cache 實際存的就是 41+OM16、4200 combos 完整）；(b) `verify_final_tar.sh` 在 WSL（nproc 16）**永遠不會觸發 tier-5** ⇒ 新增 `final48` 模式 + round 2b，證實 Linux 上 `1.295547821428148` 與 Windows 逐位相同；(c) `m67c` bundle 的 embedded-tar 成員斷言早於 `bin/` 存在（5 檔 vs 現在 6 檔）會讓 builder 直接 abort。⚠️ **更正**：換件當下以為「比對錨還指著 pre-M71」是 bug——其實 M71 那次已把 `results_shipped_m51.json` 的**內容**換成 M71 結果（舊值另存 `results_shipped_preM71.json`），所以只是**檔名過期**；現已更名 `results_shipped_m71.json`。

### 離線 GREEN、但 in-window RED（LP 家族）
- **M53 L1 品質池 ✅ offline SHIPPED**（`[[m53_l1_quality_mode]]`）：84-pool（41 base + 2 `_L1_EXTRA` + 41×K24 tier）→ 1.3176。復現 = `ICCAD_ADAPTIVE_POOL=0` + `ICCAD_L1_POOL=1` + `ICCAD_PROFILE_TIMEOUT=600`。**REFINE 向上只能走 portfolio tier**（全域 override 全池退步）；`PUSH_PASSES`/`COMPACT_ITERS` 向上逐位 no-op（early-break）。**M65**：補齊 84 池缺的兩隻 K24-extra = RED（贏處逐位 no-op、輸處全毒，期望值恰 0）。
- **M53 L2 stochastic best-of-N ❌ RED**（`[[m53_l2_stochastic_red]]`）：union oracle +0.2377% < bar 0.3%。**逐決策噪聲全域毒**（greedy 零容錯）；權重 jitter = 連續 profile 空間、8/15 重案恰 0。樣本仍作 L3 種子（GREEN）。
- **M53 L3 constraint-graph LP ✅ offline GREEN**（`[[m53_l3_lp_green]]`）：固定拓撲 + HiGHS exact 線性 HPWL → winner-only +0.76% → LP×portfolio 重選（**top-32 飽和**）1.30035 → 疊 L2 種子 **1.2978**。已探畢勿重掃：top-k>32 零貢獻、area 聯合目標僅 +0.026%、LP 2-4 pass 即 fixpoint、boundary-repair 0 可修。
- **M54 in-window LP ❌ RED**（`[[m54_lp_inwindow_red]]`）：deployable proxy-guard 形本身 GREEN（+0.891%、0 回歸）但 RF 全域淨虧；**β=0 kill test（solver 免費）最壞 cell 仍 +0.48%** ⇒ solver/C++/向量化加速軸全部 moot。復活條件：官方 median ≥~11s／RF=1.0／M62 路徑。
- **M62 break-even ⚠️ GREEN 但未動工**（`[[m62-break-even-green]]`）：M54 的「無 weak-win」只限 cap≥0.5s 網格——**cap 0.1-0.25s 的 skip gate 零加速即全 cell 弱贏**（cap=0.1 +0.06%、jitter 40/40、機速 s∈[0.5,1.5] 存活）。但 abort-only（免預測器）wins=0、s≥2 全滅 ⇒ 前置 = pre-build 時間預測器 + 機速校準。
- **組員 `80cc719` Route 1 / Route 3 ❌ 皆 RED**（2026-07-29 我方重算；`[[teammate-80cc719-route1-route3]]`）：Route 1（M71 winner + 全域 LP）1.2914 但 runtime ×2.22 ⇒ RF **+14.7~31%**；Route 3（只放開 top-12 HPWL 貢獻 block）**−0.0158%**（82% 來自 case 99 單案）+0.20s/案 ⇒ **+0.82~5.77%**。**🔑 Route 3 反向複驗 M62 的 α/β 分解**：LP 的 pairwise 建構是 **O(n²)、與幾個 unit 真的自由無關** ⇒ 縮小自由集只讓 solve 免費、build 照付，品質剩 1/50 ⇒「縮小 LP 省時」正式死。**🚨 他們的 LP run-to-run 非確定**（HiGHS `time_limit`=5.5s 在重案觸發，case 91 兩次 cost 不同）⇒ 我方未來任何 in-window LP **必須關掉 time_limit 或證明它不觸發**。唯一資產 = 離線錨 **1.2914**（不可送件）。

### quality 軸 RED（結構性）

- **M79 自建 ML 候選 Gate 0（2026-08-05；`M79_REPORT.md`、`[[m79-shape-and-knob-ceilings]]`）**——問「我方能不能自己做一顆過得了 M77 那把尺的 ML」。pre-registered bar = in-set portfolio delta ≥1.0%（OOS NET 0.30% 的 3× headroom）。**兩條路都 RED**：
  **(a) 完美形狀 = +0.099%**（12 recipe 版 +0.112%）。機制 = `constructive_m79.cpp` 的 `ICCAD_DIMS_FILE`（讀 `id w h` 覆寫 `dims[]` 並鎖住該 block，讓 `apply_safe_mib_dims`/`FREE_ASPECT`/`FREE_CLUSTER`/`FREE_ANCHORED`/`CLUSTER_ASPECT`/`MIB_ASPECT` 全跳過；off-path 600/600 逐位）。覆蓋 6110/7050 = **86.7%** 的 block、位置/排序/frame/後處理全照我們自己的 placer。**對照組（同 6 隻 recipe、不指定形狀）恰好 +0.000%（0 wins / 0 switches）⇒ 增益 100% 歸因於形狀**，沒有「多幾隻 pool」的混淆。⚠️ 事前訊號其實很強（label aspect sd(log)=0.530、只有 34% 近正方，而我們預設 `SOFT_ASPECT=1.0`；boundary code 分層後 sd(log) 幾乎不降）**——訊號強不等於分數在**。
  **(b) 逐案旋鈕 oracle = +2.025%（NET +2.023%），但不可預測**。R=128 隨機聯合抽樣、82/100 案贏、efficiency 99.7%、proxy 誤選 0。**Gate-1 預覽的 LOO 預測器（判準用 portfolio delta，不是 solo cost）只有 global +0.166% / band +0.155% / knn5 +0.091%，三個都輸給「不看案子、直接加一隻固定 profile」的 held-out +0.234%**，而且最「個人化」的 kNN 最差 ⇒ **M56 的完整重演**（winner case-idiosyncratic）。
  **🔑 教訓 A：三個 perfect-information 上界排起來就是天花板的形狀**——完美排序 +0.005%（M26）／完美位置種子 +0.001%（M68）／完美形狀 +0.099%（M79），而到 label floor 有 14.343% ⇒ **瓶頸不在餵給 packer 的決策，在 packer 的可達集合**（M27 的另一面）。**🔑 教訓 B：判準錯會給假 RED**——本探測第一版的 LOO 預測器用「訓練集平均 solo cost 最低」挑向量，報 **+0.000%**；換成 portfolio delta 就變 +0.166%。M77 的頭條（solo 與 portfolio 價值不單調相關）在**自己的工具內部**又踩了一次。**🔑 副產物見下面 GREEN 區的 M79-B′**。
- **M78 候選集合的第二條路徑（2026-08-03 RED；`M78_REPORT.md`、`[[m78-candidate-set-second-path]]`）**——假說：M71 只補了兩條 cluster 路徑中的一條（純 movable 走 `make_group_item`），**mixed（preplaced+movable）走的 `adjacent_candidates_for_block` + `pack_in_frame` anchored first-pass 仍是 pre-M71 形狀**。前件普查（`m78_antecedent_census.py`，讀 dataset 不跑 placer）**PASS**：anchored movable 佔加權 blocks **6.26 / 5.66 / 5.86%**（in-set / s1 / s2），非空案佔 62-74% 的權重，重帶 n>100 有 14/20、56/80、50/80 案非空，A1/M71 前件比 **0.30-0.36×** 三語料一致。⚠️ 但「有 A1 前件而 M71 前件為空」只佔 **0.00/2.55/3.59%** ⇒ 兩者**案子層面高度重疊**。**六個旗標（全部只進 `constructive_m78.cpp/.exe`，出貨 exe md5 不動 ⇒ 三顆 cache 不失效）× 11 個 arm，10 個變差、1 個變好**：唯一贏的 `ICCAD_M78_ANCH_CROSS`（跨 rect 交叉候選：x 取自一個 rect 的面、y 取自另一個——出貨的候選集合 x/y 永遠取自同一個 rect）in-set **−0.183%**（16c）/ **−0.213%**（48c）、**OOS 240@48c −0.1604%**（33 好 40 壞）、wall ×1.07 ⇒ 品質只有 bar（0.30%）的 **54%** 且 wall 是扣分 ⇒ **RED**。tier 形式也否決：2-way per-case oracle 上界 **+0.3841%**（overlay 只 realize 41.8%），但要 41 隻 twin 才拿得到，而 48c wall = max-setter、twin 慢 7% ⇒ max-setter 的 twin 一進池就是 `1.07^0.3` = **ΔRF +2.04%**；避開就得 in-sample 挑 source set（M76 已量轉移率 ≈5%），對照 M76 escape tier 4 隻 source 只 realize +0.101~0.107%。**🔑 教訓 A：「加候選」預設是有害的**——完全相同的交叉機制在 anchored 是 −0.18%、在泛用 `item_candidates` 是 **+0.36%**，中心對齊槽兩邊都是正的（+0.29 / +0.40%）⇒ **出貨的候選集合不是貧乏而是調過的**，greedy 的 `bbox_area_with` 短視，多給局部更好的位置反而做出全域更差的選擇。**🔑 教訓 B：M71 的排序增益搬不過來**——`ord1`（corner-first）**恰 0**、`ord2/ord3` 只動 3 案（0.92% 權重）、`ord4`（拿掉 boundary 優先序的對照組）**+1.069%** ⇒ anchored 排序軸整條封卷。M71 之所以能從 7 種成員順序拿分，是因為每種順序餵出 5 種**內部版型**再用 layout key 挑；anchored 成員逐一直接放進 frame，**沒有內部版型可排列**。**🔑 教訓 C：這是「機制真但太小」的 RED，不是「前件空」或「效果被吸收」**——in-set→OOS **轉移率 76%**（機制的轉移率，遠高於 M76 那個 source-set 的 5%），movers 精確落在普查預測的案子上（90/86/88/93/95）。

- **M76 組員 M73 knob-OFF escape tier（2026-08-01 RED；`M76_REPORT.md`、`[[m76-escape-tier-red]]`）**——組員 `7403758` 的機制：append knob-off 的 host 副本、`_solve_impl` 對那些 index 故意不套 `_m71_env()`，讓被 M71 弄壞的案子逃生。已完整移植（預設 off、gate-off 逐位 `1.293461035226291` 0 movers）。**判定 RED**：三個變體（組員集全帶 / 組員集 n>100 / 我方 M74 下重推的 `(21,23,2,22)` n>100）在**評分機的 48 核池形狀**下 OOS 240 全部只有 **+0.101~0.107%**，bar 是 0.30%；48c ΔRF 代價 +0.020~0.088% ⇒ NET 只剩 +0.02~0.09%。**🔑 主因：escape tier 與 tier-5 是替代品**——OOS shipped 基準本身從 16 核形狀的 1.576749 掉到 48 核形狀的 **1.555855**（−1.325% = tier-5 的 OOS 價值），tier-5 放回 n>100 的那 22 隻 knob-ON profile 已經救掉 escape 原本要救的案子，同一批分數不能算兩次。**組員的 +0.288% 是低核數字**（12 核 + 他們沒有 tier-5，看不到抵銷）。**🔑 教訓 A：in-sample 優勢轉移率 ≈5%**——`m73x` vs `m73big` 是同機制同 gate 只換來源集的乾淨對照，in-set 差 +0.127pp、OOS 只差 **+0.006pp** ⇒ 來源集不值得用 in-sample 貪婪挑。**🔑 教訓 B：OOS 也要挑對池形狀**——in-set 100 在 16/48 兩種形狀下**逐位相同**（1.289345、5 movers），OOS 卻差 **2.7 倍**（+0.294%→+0.107%）⇒ 任何與 adaptive tier 有交互作用的機制，OOS 必須用 `m67_oos_probe.py --force-cores 48`。副產物：組員未解的「knob-off 逃生口會不會在 48 核變成 max-setter」＝**否定**（重帶 `dt_esc/dt_on` p50 **0.906~0.934**，knob-off 比 knob-on 更快 ⇒ (110,inf] ΔRF 恰 **+0.000%**）；他們的 12 核 mid wall +3.8~9% 在 48 核只有 **+0.064%**（又一個 tier-3 形狀的誤判），但「mid 不要開」的結論仍成立，理由是 mid 只多 +0.014pp 品質卻多 +0.068pp wall。⚠️ 條件依賴：若評分機有效並行度 <40，tier-5 不觸發、抵銷消失、escape 回到 +0.29% ⇒ 這條 RED 與 tier-5 共用同一個賭注。

- **M75 M71 剩下四個旗標（2026-07-31 全 RED；`M75_REPORT.md`、`[[m75-m71-residual-knobs-red]]`）**——全域 overlay 形式（**不是** M72 的 pool tier），OOS 240 判準、bar +0.3%。錨 OOS `1.576749` / in-set `1.293461035`。**CORNER = 恰 0.0000%、REPACK ≈0**（in-set 100 + OOS 240 共 340 案 × 全 pool，幾乎零個 profile 的 binary 輸出改變。⚠️ **2026-08-03 M78 更正**：REPACK 原記「恰 0」是 `_m75_liveness()` 餵 `target_positions=None`、把 preplaced 全刪掉造成的——那讓 mixed cluster 一個都不存在、anchored 旗標前件恆空。**用正確輸入重測 = 1/3500（in-set 100×35，只有 case 75）**，判定不變、措辭要改；CORNER 重測仍是 **0/3500**，原判定原封不動。見 `[[gate-inputs-must-match-deployment]]`）；**SLIDE = 恰 0**（52 個 (case,profile) 有動，但**全部在 n≤55 的小案且全輸 proxy argmin** ⇒ portfolio movers 0/240）；**PERMUTE = −0.0111% OOS**（4 movers：n=73 **−15.65%**、n=76 −1.40%、n=60 −0.86%，但 n=104 **+4.58%** 在 `exp(n/12)` 下全吃掉）。**四旗標全聯集逐位等於 PERMUTE**——已實測：以 M71+PERMUTE+SLIDE 為底再加 CORNER/REPACK/兩者，**7900 個 pair 改變數 0** ⇒ 15 個 arm 的矩陣合法塌縮。⚠️ **前件全部非空**（REPACK 216/240 案、SLIDE 198、PERMUTE 182、CORNER 67）⇒ 這**不是** M60 那種前件空集，而是「前件活、效果被吸收」：REPACK 的 ±9000 偏置對既有 `BP_W*bp`=30000 幾乎恆保序；CORNER 的候選從未嚴格勝出；SLIDE 的三重 guard 在 compaction 後緊密版圖上幾乎必失敗。**🔑 教訓：in-sample 不只會藏住 OOS 差距，還會給相反符號**——PERMUTE 的 in-set 是 **+0.0104%（正）**、OOS **−0.0111%（負）**，只看 local100 會判成弱 GREEN。**🔑 方法論：liveness 不可用 portfolio 輸出判**（旗標能改候選卻改不動 proxy argmin ⇒ 四個都報「零差異」的假 RED），要用 **per-profile binary 輸出**；而且一旦證明某案全 pool profile 逐位相同，該案 portfolio 就可證明不變 ⇒ 只解活案即得**精確**加權 delta，比抽樣 arm 又快又嚴格（本輪 ~71000 runs / 0 失敗，取代 4×15 分鐘 arm）。**組員的 +1.287%~+1.531% 是整包 profile 配方的增益，不是這些旗標的**。
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
- **`make_submission.py verify` 不要丟 background**：harness 可能回報「完成」但 python 子行程仍在跑，留下的 `constructive.exe`（Windows 端會 on-site compile）鎖住 `build_submission/verify/` ⇒ 下次 `shutil.rmtree` 噴 `PermissionError [WinError 5]`。清法 = `Get-CimInstance Win32_Process` 找 `make_submission`/`iccad2026_evaluate` 的 PID 砍掉、再砍 `constructive`，然後刪 `verify/`。**M73 踩過一次，還誤砍了活著那輪的子行程**
- **tar md5 不可重現**（gzip 內嵌 mtime）：`m67c_make_linux_bundle.py` 會**重跑 `make_submission.stage()`** ⇒ 建 bundle 後 tar 就換一顆 md5。要送哪顆就對哪顆跑 `verify`，身分比對一律用 `op_wrapper.py` 的 md5

## env 旋鈕

### `constructive.cpp`
- 預設：`ICCAD_BP_WEIGHT`=30000、`ICCAD_WIRE_MULT`=×1、`ICCAD_ANCHOR_W`=0.10、`ICCAD_LR_ASPECT`/`ICCAD_TB_ASPECT`=2.50/0.40
- 關後處理：`NO_COMPACT`/`NO_REFINE`/`NO_PUSH`/`NO_BND_PUSH`/`NO_SWAP`/`NO_JUMP`；`PUSH_PASSES=N`、`COMPACT_ITERS=N`、`REFINE_ITERS=N`
- pack-order：`WIRE_TIEBREAK`、`WIRE_BFS`、`BFS_PIN`、`ORDER_SWAP=K`、`ORDER_MOVE=K`、`GUIDE_MED`
- **free-aspect 六子軸**（全 gated、off=bit-identical）：`FREE_ASPECT`（single interior，`FREE_RATIOS`）／`LR_ASPECT`+`TB_ASPECT`（decoupled **uniform** boundary）／`CLUSTER_ASPECT=r`（uniform reshape 純 movable interior 成員）／`FREE_CLUSTER`+`FREE_CLUSTER_RATIOS`（per-member，layout-key 仲裁、build-time 免抬牆）／`FREE_ANCHORED`+`FREE_ANCHORED_RATIOS`+`FREE_ANCHORED_BND`（mixed cluster 成員在 wall-attach 搜 aspect）／`MIB_ASPECT=r`
- **🏆 M71 六旗標**（全部預設 0 = binary 單獨逐位不變）：`ICCAD_CLUSTER_BND_EXPOSE`、`ICCAD_CLUSTER_BND_EDGE_PACK`（wrapper `_m71_env()` 只開這兩個）；**另外四個 = M75 已量、全 RED，勿再試** = `CLUSTER_BND_CORNER`／`ANCHORED_BND_REPACK`（恰 0.0000%，340 案零 profile 輸出改變）、`HPWL_SAFE_CLUSTER_SLIDE`（恰 0，只在 n≤55 動且全輸 argmin）、`CLUSTER_BND_PERMUTE`（**−0.0111% OOS**）。`ICCAD_M71=0` 逐位還原 pre-M71
- 死路（code 保留 gated off）：`BFS_NORM`、`CLUSTER_ORD=1/2`、`REFRAME`、`FREE_CLUSTER_BND`
- 離線探測（永不 ship）：`ORDER_FILE`+`ORDER_GLOBAL`（oracle-perm）、`ML_ANCHOR`（M68，在 `constructive_m68.cpp`）、**`DIMS_FILE`（M79，在 `constructive_m79.cpp`：讀 `id w h` 覆寫 `dims[]` 並鎖住該 block，所有 reshape 路徑跳過它；off-path 600/600 逐位）**
- `ICCAD_CONSTRUCTIVE_SINGLE=1` 退單 base profile

### `optimizer_constructive.py`（wrapper）
- **`_pool_indices()` 的 adaptive 砍法**（`ICCAD_ADAPTIVE_POOL`，預設 **1**；`=0` 還原 full 41-prof + full REFINE）：
  - M41 砍 swap（`ICCAD_ADAPTIVE_N=K` 只砍 n>K）
  - M42 砍 `_BIG_REDUNDANT_IDX`（22 隻）當 `n > ICCAD_ADAPTIVE_FREE_N`（預設 100）
  - M45 tier-3 砍 `_M45_BAND_DROP`（**M74：15 隻**，60<n≤100；`ICCAD_ADAPTIVE_BAND=0` 關）
    ⚠️ **M74 起 tier-3 也是 cores-gated**：只在 `_effective_cores() ≤ _M45_MID_CORES_MAX`（=**16**）才開。高核上 mid 帶是 max-setter-bound（c\* max 15.2），剪了買不到 wall 卻付 OOS 品質 −0.702%。unknown→9999 ⇒ tier 關 = 滿 pool = 安全側
  - M45 tier-4 低核：`_effective_cores() ≤ 8` 時砍 `_M45_LOWCORE_DROP`
  - **M67-F tier-5**：`_effective_cores_hi() ≥ 40` 時**跳過** `_BIG_REDUNDANT_IDX`（`ICCAD_M67F_TIER5=0` 關）
  - **🏆 M80 knob-cloud tier**（`ICCAD_M80_TIER`，**預設 1**；`=0` 是 kill switch）：
    `_effective_cores_hi() ≥ _M80_CORES_MIN`(**40**) 時把 `_M80_EXTRA`（8 隻，idx **86-93**）
    掛進池。`ICCAD_M80_MIN_N`（預設 0）是帶別 gate。gate 在 `ADAPTIVE_POOL=0` early-return
    **之前**讀（同 M72/M76 紀律）。**M80 索引會拿到 `_m71_env()`**（cloud 就是在它下面量的）
    ——這點與 escape tier 正好相反。⚠️ **`_M80_CORES_MIN` 與 `_M67F_CORES_MIN` 同為 40**
    ⇒ 39c→40c 之間**兩個 tier 一起翻**，任何「只驗 M80」的檢查都要先把 tier-5 釘掉
    （`m80_tier_gate._ISOLATE` / `m67g_tier5_gate._ISOLATE` 是對稱的兩把鎖；M80 開發時
    `m67_oos_probe` 的 cores-gate 檢查就是漏了這個而報 39c→13 / 40c→43 的假 FAIL）
  - **M76 escape tier**（`ICCAD_M73_ESCAPE`，預設 **0**；ship RED 見 ledger）：`_M73_ESCAPE` 是 41 隻 host 的 knob-off 副本（idx 45-85），`ICCAD_M73_SRC`（預設 `_M73_SRC=(2,22,23,25)`）選子集、`ICCAD_M73_MIN_N`（預設 0）做帶別 gate。gate 與 M72 一樣在 `ADAPTIVE_POOL=0` early-return **之前**讀。escape 索引的 overlay 由 `_profile_env(i,n)` 決定 —— **不套 `_m71_env()`**，那就是機制本體
  - `ICCAD_ADAPTIVE_CORES=N` 強制核數（tier-4/tier-5/M50 共用；`=48` 可在本機讓 tier-5 真觸發）
  - ⚠️ **`_effective_cores_hi()`（unknown→0）是高核 tier 專用，不可與 `_effective_cores()`（unknown→9999）混用**
- **`_band_env()` REFINE band-cut**：n>100 疊 `REFINE_ITERS=4`；60<n≤100 疊 **`=6`（M74：8→6）**（cores≤8 改 `=4`）。`ICCAD_ADAPTIVE_REFINE=0` 關
- **offline 旋鈕（永不送件）**：`ICCAD_L1_POOL=1`（84 池，須配 `ADAPTIVE_POOL=0` + `PROFILE_TIMEOUT=600`）、`ICCAD_M67F_RESTORE=1`（只跳過 M42+tier-3，量 θ 用）、**`ICCAD_M55_POOL=1`**（M72：`_M55_EXTRA` 4 隻 boundary-aware cluster profile 掛進池，idx 41-44；ship RED 見 ledger。gate 在 `ADAPTIVE_POOL=0` early-return **之前**讀，故 full-pool 路徑不會漏）
- **M48**：`ICCAD_CXX=編譯器` 插到編譯鏈最前；每次編譯成功都須過 `_binary_runs()` 1-block smoke
- ⚠️ 改 `_PROFILES` **或重編 constructive.exe** 後必須：`profile_audit.py base` + `profile_audit.py ship` 重建兩顆 cache → `ICCAD_REGEN=1 rf_score_model.py` 重算全部 drop 常數（貼回後不帶 REGEN 再跑一次要全綠）→ `m49_refine_probe.py` 三 gate → `m67g_tier5_gate.py`。
  自 M74 起 cache 簽章已釘 **exe md5 + overlay 常數**，所以忘了重建會**明確報錯**而不是靜默用舊資料

## 工具（全部永不 ship）

- **`regression_suite.py`** — 送件前一鍵**八**項 gate（m48 四 phase → rf_score_model asserts → m49 三 variant → m47b proxy 等價 → m67g tier-5 → **m80 tier 身分**），子行程先剝 `ICCAD_*`，~13-15 分鐘
- **`m80_tier_probe.py`** `build|score|selftest|inset` — M80 的判定工具。**重用 `m77_oos_audit.pkl` 唯讀**（2 樣本 × 240 案 × 35 隻的 positions+dt+proxy），只自己跑 K 隻新向量（key `(sample, case, profile-hash)` ⇒ 加大 K 只付新向量的錢）。`score` 出 **K=0..Kmax 的 prefix 曲線**（quality / dRF@48c / **NET**），不是單一數字；`selftest` 的 K=0 必須逐位重現 m77 的 shipped 總分（已 PASS，`1.555854672`）；`inset` 從 `m79_knob_cloud.pkl`+`audit_cache_ship.pkl` 出 in-set 曲線與 **@48c 與 @12c 兩份 dRF**，不跑任何 solver。⚠️ `--cores 48` 是預設也是唯一有意義的形狀
- **`m80_tier_gate.py`** — M80 池身分閘（regression_suite 第八項）：V1 惰性（auto/4/8/12/16/24/32/39 核）、V2 blast radius（40/48/96 核加的恰是 `_M80_IDX`、REFINE 不動、`MIN_N` 帶別 gate）、**V3 出貨前綴 == HEAD**（這條就是「四顆離線 cache 仍有效」的證明）、V4 fail-closed、V5 `_M80_EXTRA` 逐字 == `m80_vectors.json`、V6 可達性（無 ORDER_SWAP/MOVE）+ M71 overlay。⚠️ V3 要 `git show HEAD:` ⇒ **wrapper 改動要先 commit 再跑**
- **`m80_vectors.json`** — 12 隻的貪婪順序（出貨取前 8）+ seed/R/order。**K 隻向量唯一的機器可讀來源**：`build_cloud()` 雖然是 seeded，但輸出**依賴出貨前綴**，沒有這個檔「#100」就是會漂移的指標
- **`make_submission.py`** `stage|verify|all` — 產 `build_submission/cadc1075/`（6 檔）+ tar，verify = 官方指令 100 案逐位比對
- **`m67c_make_linux_bundle.py`** + WSL `run_all.sh` / `verify_final_tar.sh` — Linux 四關（build+smoke / m48 opwrapper / 官方 100 案 bundled-first 逐位 / 破壞 binary 落編譯鏈）。`m67c_tier3.py` 模式：`t3` / `t4` / `final <tar>` / **`final48 <tar>`（M73 新增：強制 `ICCAD_ADAPTIVE_CORES=48` 讓 tier-5 在 WSL 也跑得到，錨 `results_M73_cores48.json`）**；`verify_final_tar.sh` 現在兩輪都跑，末行 `VERIFY_FINAL_TAR: ALL PASS`。⚠️ 換 bundle 要**整包重傳**（md5 對不上就是舊的，grep `final48` 可秒判）
- **`m48_coldstart_dryrun.py`** — 冷啟動四 phase（含 `opwrapper` variant）
- **`rf_score_model.py`** — RF 投影 + M42/M45 drop 常數 regen + drift asserts（讀 **`audit_cache_ship.pkl`**；`ICCAD_REGEN=1` 把四個 drift assert 降級成 warning，讓一次跑就印出全部三組建議常數）；**`m67e_rf48.py`** — 48c 投影（`gate0/calib/fit/project/report`，投影看 `restoreIdx`）
- **`m49_refine_probe.py`** `trace|variant K [big|mid]` — REFINE band gate；**`m67g_tier5_gate.py`** — tier-5 池身分閘（V1 基準用 kill switch，不可比 HEAD）
- **`m67_oos_probe.py`** — OOS 泛化（`gate0/run/report/ref/pool0/restore`，`--pool0-lo/-hi` 選帶）；`m67_oos_cache.pkl`。⚠️ **錨已於 M75 更新為 `results_M74_default.json` / `IN_SET_TOTAL=1.293461035226291`**（原本指著檔名誤導的 `results_shipped_m51.json`＝M71 內容，且 `IN_SET_TOTAL` 還是 pre-M71，不修會把 M74 自己的 14 個 movers 報成 arm 的）。🚨 **跑完變體 sweep 要把 live cache 還原成預設組態**——M75 開場就撞到 tree 上的 `m67_oos_cache.pkl` 是某個 M74 變體殘留、sig 對不上，`gate0` 一載入就清空 240 案（從 `m67_oos_cache.pkl.M74k6` 還原）。**arms = `pool`（M42+tier-3 還原）/`refine`/`m55`（M72 tier + M71 全域 off = 組員原形）/`m55x`（tier 疊在 M71 上）/ M75 的 15 個 `m71*` 純 C++ 旗標 arm（4 單 + 6 pair + 4 triple + 1 union，全 RED）/ **M76 的 `m73`（組員集全帶）`m73big`（組員集 n>100）`m73x`（我方集 n>100）**（全 RED）**／**M80 的 `m80`（tier 全帶）`m80big`（n>100）`m80off`（GREEN，已 ship）**。🚨 **M80 起 arm 有方向性**：`_shipped_pool()` 是用「剝掉 gate key 取預設」決定基準，所以 tier 一旦預設 ON，`m80` 就變成 no-op，要用 `m80off` 才量得到東西（Gate A 用 `sign` 自動判方向）——與 `m67g` 對 tier-5 記的「tier 一 commit，跟 HEAD 比就是空操作，kill switch 才是永久不變式」同一課。arm 名不進 `_sig()` ⇒ 加 arm 不會作廢 cache；無 `full` 端點時自動退成 A/B 報告並 dump `results_M72_ab_<arm>_<lo>_<hi>.json`。⚠️ **M76 起 `_sig()` 改錨出貨前綴 `_PROFILES[:_M55_BASE_LEN]`**（先前含整個 `_PROFILES`，所以每次移植 gated-off tier 都會白白清空 240 案；備份 `m67_oos_cache.pkl.preM76`）。🚨 **`--force-cores N`（M76 新增）：把 `ICCAD_ADAPTIVE_CORES` 在 ICCAD_* 剝除之後重新塞回去，用評分機的池形狀跑 OOS，並自動改用 `m67_oos_cache_c48.pkl`**——`_sig()` 不含核數，共用同一顆 cache 會靜默重用錯形狀的解。**M76 證明形狀差 2.7 倍，所以任何與 adaptive tier 交互的機制都必須跑這個**；48 形狀要先 `run --force-cores 48` 建 shipped 端點（`restore` 只解 arm 側）
- **`m77_ml_candidate_probe.py`** `score <results.json> | selftest` — **外部候選（ML placer）值多少**。輸入 = 官方 results json（任何 optimizer 跑一次就有），把它的逐案 positions 當成第 42 隻候選接進 41 隻池、proxy 仲裁，輸出 **portfolio delta（= gate，bar 0.05%）／oracle delta／selection efficiency／dRF@48c**。`selftest` 把 portfolio 自己的輸出餵回去必須恰好值 0（已 PASS）。🔑 **建這支的理由**：組員的 ML kill gate 用 **ML-only solo total**（rung 2 < 1.6），但部署形態是 proxy 仲裁的 portfolio，兩者**不單調相關**——實測反例：M74 自己的輸出 solo **1.2935**（最好）portfolio 價值 **恰 0**，knob-off portfolio solo **1.3378**（差 3.4%）portfolio 價值 **+0.340%**。⚠️ `--dt` 要給模型自己的推論時間，json 的 `runtime_seconds` 是整個 solve 的 wall（工具會警告）。**這支只管 in-set 100；OOS 240 走 `m77_oos_probe.py`**。回報全文 `M77_ML_GATE_NOTE.md`
- **`m77_oos_probe.py`** `manifest|build|selftest|selfdump|score` — **M77 的 OOS 半邊**（2026-08-02）。為什麼要它：M76 量到 **in-sample 優勢轉移率 ≈5%**，且 OOS 數字**依池形狀而變 2.7 倍** ⇒ ML 的 ship 判定必須在 OOS ×`--cores 48` 上做。**兩份樣本**：`s1`=M67-D 那批（worker_0..9、seed 67、per_n 2 / heavy 4），manifest 會**斷言 240 key 與 `m67_oos_cache.pkl` symdiff=0** ⇒ 與所有歷史 OOS 數字同尺；`s2`=**worker_10..19 的 disjoint 240**，因為 s1 抽自 `floorset_lite`＝組員的訓練語料，**對 ML 候選只有 s2 是真 OOS**（報告會印警告橫幅）。快取 `m77_oos_audit.pkl`（2×240×**35** 隻 = 12/48 核池的聯集；swap profile 被 M41 永久濾掉所以不跑）存 positions+dt+proxy，**以 `(case_key, profile)` 為 key ⇒ 加第三份樣本只付新案的錢**；簽章釘 exe md5 + 出貨前綴 + REFINE band（不含 drop 常數，那些只影響 score 時選池）。🔑 **一顆快取服務兩種核數**——實測 `_band_env(n)` 在 12/16/48 核完全相同，只有 `_pool_indices()` 不同。**判準 NET = portfolio delta − dRF@48c ≥ 0.30%**（M75/M76 的 OOS bar），聚合用 `m67_oos_probe._per_n_total`。✅ **驗過**：s1 240/240 案的 winner positions 與真 wrapper **逐位相同**（12 核形狀 `1.576748536`、48 核形狀 `1.555854672`，與 M76 的錨一致）、s1/s2 零值自檢皆 +0.000%、錯位 key 直接退出。⚠️ 為什麼不能用 2-way 近似：selector 的 `hmin` 是**整池**的 min HPWL，ML 候選壓低 hmin 會重排整池。**📊 副產物**：M74 在 s2 的分數 = 12 核形狀 `1.586912` / 48 核形狀 `1.557814` ⇒ (a) **s2 只比 s1 難 +0.126%**（48 形狀）⇒ 兩語料難度相當，s1↔s2 的落差可直接當記憶效應的量測；(b) **tier-5 在全新語料上獨立複現且更值錢**（s2 −1.833% vs s1 −1.325%）——M67-F 的賭注第一次在 s1 以外被驗證
- **`m79_shape_oracle_probe.py`** `coverage|calib|scout|oracle|control|offpath` — M79 G0-A/G0-D。`calib` 把 fp_sol verbatim 寫成 results json（校準列，+14.410%）；`oracle` 用 `constructive_m79.exe` 的 `ICCAD_DIMS_FILE` 指定 label 形狀（86.7% 覆蓋）；**`control` 是同 recipe 不指定形狀的對照組，必須恰好 +0.000%**（若不是就代表 recipe 子集本身在貢獻，歸因就壞了）；`offpath` 是 probe binary 的逐位閘。recipe 集用 `M79_RECIPES=0,2,6,...` 覆寫，快取按 `mode+sig[:8]` 分槽，換 recipe 不會蓋掉舊量測
- **`m79_knob_cloud_probe.py`** `run [R] | greedy [R] [KMAX] | loo [R] [KMAX] | dump [R] [order]` — M79 G0-B / **M80 的向量來源**。`run` 抽 R 個旋鈕向量 × 100 案；**`greedy` 是真正有價值的那個**（單一最佳新向量 + 貪婪固定 profile 曲線，**收尾會寫 `m80_vectors.json`**）；`loo` 是 LOO 逐案預測器 + **greedy 集的 5-fold CV**；**`dump` 用明確的 order 重生向量**（預設 order 是 M79 那 8 隻 ⇒ `dump 128` 就是 prefix-stability 驗收）。快取 `m79_knob_cloud.pkl` 的 key = **`(case, profile-hash)`** ⇒ 加大 R 只付新向量的錢；簽章釘 exe md5 + overlay 常數（不含 cloud）。**M80 起 `KMAX` 可調（預設 12）**——原本硬編 8，但 held-out 曲線在第 8 步還在漲 +0.198pp，停在 8 只會低估。⚠️ **任何預測器的判準都要用 portfolio delta，不可用 mean solo cost**（第一版就是這樣報出假的 +0.000%）。⚠️ **`build_cloud` 依賴出貨前綴** ⇒ 前綴一動，所有「#100」就指向別的向量（`m80_vectors.json` + `m80_tier_gate` V5 是防呆）
- **`m79_bar_spec.py`** — 把 ship bar 反解成逐案規格：配額表（只贏 top-20 要每案 0.37% 才湊到 0.30%）、headroom 表（到 label floor 共 14.343%，top-20 佔 11.723%）、**dt 預算表**（各帶 min tmax 0.16/0.79/1.37s；均勻 dt=1.0s 只要 +0.119%，dt=2.0s 就 +3.253%）
- **`m76_escape_probe.py`** `oracle|wall|derive|score|report` — M76 離線工具，把 `audit_cache_ship.pkl`（knob-ON）× `audit_cache_esc.pkl`（knob-OFF）合併成單一 index 空間（`ESC0+k` = host k 的 knob-off 雙胞胎），可**精確**模擬任何 escape 來源集的 portfolio（三個端點對真 eval 逐位驗過）。`wall` 給 48c/12c 的逐案 `ΔRF=(t_new/t_old)^0.3`。⚠️ 池一律走 `oc._pool_indices()`，**不可自己拼** —— M41 的 swap 過濾是**依內容**的，也會濾掉 swap profile 的 escape 副本，手拼會選出 wrapper 根本不會跑的來源集
- **`profile_audit.py [base|ship|esc]`** — **M74 起兩個模式、M76 起三個**：`base`→`audit_cache.pkl`（M71 env + REFINE=12，給 m49 的 K=12 control）、`ship`→`audit_cache_ship.pkl`（再疊 `_band_env(n)`，給 pool drop 推導）、**`esc`→`audit_cache_esc.pkl`（`_band_env(n)` 但 M71 旗標關 = escape 索引實際跑的組態）**；`esc` 跑完會對 ship cache 做交叉檢查，兩者若完全相同就 abort（代表 overlay 沒進 binary）。約 8-11 分／顆，**必須序列跑**（dt 是量測值）、**`profile_vs_portfolio.py KEY=VAL`**（新 profile 增益，bar 0.05%）、`analyze_constructive.py`、`portfolio_ceiling.py`、`rh_sweep.py`、`proxy_analysis.py`（27 個工具依賴，勿刪）
- **`m53_diff_results.py`** — 兩份官方 results json 的總分/加權 delta/逐案 movers。錨：`results_L1_final.json`、`results_L3_port_top32_area.json`、**`results_shipped_m71.json`（= 出貨錨 1.305389893450635，`make_submission.verify` / `m67c` T3 都比這顆）**、`results_M73_cores48.json`（48c/tier-5 錨 1.295547821428148）、**`results_M74_default.json`（1.293461 = 現在 tree 的分數）**、`results_M74_cores48.json`（48c 同值）、`results_M74_pool0.json`（1.2929 天花板）、`results_shipped_preM71.json`（1.3265）、`results_shipped_m51.json`（**檔名誤導**：內容已是 M71，留著相容舊 probe）。⚠️ 這些錨檔**未進 git**（沿用舊慣例），但 gate 依賴它們——換機器要一起帶。
- **RED 存檔 probe（勿重跑求更好的數字）**：`m52_phase0_probe.py`、`m53_l2_probe.py`、`m53_l3_probe.py`、`m54_lp_rf_model.py`、`m55_dropset_cv.py`、`m56_percase_oracle.py`、`m57`~`m61`（各配 `constructive_mXX.cpp`）、`m62_break_even.py`、`m63_vio_bound.py`、`m64_flip_probe.py`、`m65_l1_cell.py`、`m66_equiv_cv.py`、`m67f_contention_probe.py`、`m68_ml_seed_probe.py`、`oracle_perm_probe.py`、`dbg_seqpair.py`、`recon_slice_probe.py`、`tree_decode_probe.py`

## 檔案結構（要點）
- `constructive.cpp` 🏆 — placer，含 M9-M37 + M46 hot-path exact + **M71 六旗標**
- `optimizer_constructive.py` 🏆 — 41-prof portfolio + shapely proxy(_RH=1.4) + `_pool_indices()` **六**階 tier（M41/M42/M45×2/M67-F tier-5/**M80**）+ `_band_env()` + `_m71_env()` + `_m80_active()` + M48 三層安全網
- `constructive_m71.cpp` — 組員合併前的參照本（其餘 `constructive_mXX.cpp` 為各 probe 儀裝副本）
- `constructive_m79.cpp` — M79 probe：`ICCAD_DIMS_FILE` 形狀指定（gated，off = 逐位相同）
- `m80_vectors.json` 🏆 — M80 的 8+4 隻向量，`_M80_EXTRA` 的機器可讀來源（gate V5 逐字比對）
- `bin/constructive_linux` — Beta 包 bundled binary（`-O3 -static-libstdc++ -static-libgcc`、無 `-march`）
- `iccad2026contest/iccad2026_evaluate.py` — 官方評估腳本
- `AGENTS.md` — 給 codex 的 CLAUDE.md 副本（2026-07-22 快照，已過時、未追蹤）

## ML 現況

**我方五種插入點全 RED**（見戰略結論 6-7）。⚠️ 本環境 torch 是 **2.11.0+cpu、無 CUDA、16 核**；**使用者 2026-08-05 裁示：小模型可在本機 CPU 訓，大模型仍要上 GPU 機；訓練訊號一律 self-supervised，完全禁止用 fp_sol 當監督**（離線 oracle 探測用 label 不受限，那和 M26/M68/M79 同類）。

**🚨 2026-08-05：「我方自己做一顆」也判死了（M79 Gate 0）。** 兩條路都量過：
per-block 形狀的 perfect-information 上界只有 **+0.099%**；逐案預測 profile 的 oracle
有 **+2.03%**，但 LOO 預測器只拿到 **+0.09~0.17%**，**輸給不看案子的固定 profile**
（held-out +0.234%）⇒ 這是 M56 的完整重演。⚠️ **但同一批探測撞到一條古典增益**
（隨機聯合抽樣的固定 profile，held-out NET +0.655%）⇒ 見「下一步 0（M80）」。

**2026-08-01 新局面**：組員取得學長 `cadc1106` 的截圖——同一批 100 validation cases、**Total 1.1747 / Avg 1.2088 / 100 feasible / 0.81s**，比我方 M74 好 **−9.2%**，落在 `fp_sol` verbatim floor（1.1079）之上所以**不違反資訊理論下界**（無法從數字判定是否 label 洩漏；只有截圖，無程式碼、無結果檔）。組員據此裁示**全力轉 ML-as-placer**（`b7460d3`「目前計畫.md」，21 天、硬分叉日 8/8、GPU 獨占、RF 量測暫停）。他們的論點：我方 RED 全在 advisor 軸（完美輸入也只值 0.005~3%），而唯一量過的 as-placer（IL68）只訓練了 10 萬次樣本呈現 ≈ 學長的 1/4000，曲線還在降。

**🚨 2026-08-03 更新：ML-as-placer 由組員自己判死，建議提前分叉回古典線。**
組員 `b86a02c`（branch `il-route-update-2026-07-31`，08-01）跑了 **oracle 天花板測試**
（`oracle_pack_ceiling.py`，讀 label 的離線量測、不可出貨）：**餵給他們的 packer 一份滿分
答案，整條管線也只拿 3.7518**（99/100 feasible）；對照組 verbatim **1.1079 逐位相同** ⇒
抽取正確。他們 07-31 預註冊的 gate 是 `rung 2 ≤ 1.585`，**天花板是門檻的 2.4 倍** ⇒
**用證明而非實驗判 FAIL**，rung 1（18h）與 rung 2（3.8 天）不必跑。
再拆一層排除實作缺口：把 frame 換成該案真實 bbox、其餘不動 ⇒ 零 slack 時成功的 42 案
**每案 1.017（優於 fp_sol verbatim）但 58% 擺不完**；任何讓它擺得完的 slack 都毀掉品質
（×1.15 擺成 98/100 但 hpwl 0.4511）⇒ **懸崖不是斜坡，正是 M52「零容錯帶」在擺放程序上的
體現**。他們三處自我更正：「packer 結構性失真」錯（零 slack 下幾乎逐位重現）、「frame 是
元凶」錯（只改 frame 最佳 3.7913）、「向量化 13×」錯（實測端到端只有 1.6×）。
⇒ **待辦「ML 候選判定」沒有候選會進來**，`m77_*` 兩支工具轉為**待命**（機制仍正確，
若日後拿到學長 `cadc1106` 的產物可直接用）。未解決：對手 1.1747 仍無解釋（已索取產物、
未回覆）、訓練慢 17× 真因未定位、他們仍未本地複驗我方 M73/M74。

**我方在此的角色 = 判定，不是訓練**（環境禁訓）。工具 = `m77_ml_candidate_probe.py`（in-set 100）+ **`m77_oos_probe.py`（OOS 240，2026-08-02 接好）**，回報 = `M77_ML_GATE_NOTE.md`：
1. **他們的 kill gate 度量錯了**——用 ML-only solo total，但部署是 proxy 仲裁的 portfolio，兩者不單調相關（實測反例見工具區）。正確 gate = portfolio delta（in-set bar 0.05%、**OOS NET bar 0.30%**）。
2. **他們的 fallback 錨過期 2.5%**（寫 1.3265 = M67-G，實際 tree 上是 1.2935）。
3. **proxy 在異質候選上是 oracle-perfect**（M76 full-union 逐位等於 2-way oracle；M77 驗證 17/17 撈到、efficiency 100.0%）⇒ ML 不需要「總是」贏，只要「有時候」贏。
4. ⚠️ 他們若要量 OOS，**必須用評分機的池形狀**（M76：in-set 兩形狀逐位相同、OOS 差 2.7 倍）——`m77_oos_probe.py --cores 48` 已內建。
5. 🚨 **判定要用 `s2`**：歷史的 OOS 240（s1）抽自 `floorset_lite` worker_0..9 ＝ 他們的**訓練語料**，對 ML 候選是樣本內。s2 = worker_10..19 的 disjoint 240，與 s1 交集 0。**兩份都跑，差距本身就是記憶效應的量測**。
