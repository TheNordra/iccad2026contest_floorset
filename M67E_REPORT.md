# M67-E — 48c RF 投影（alpha 錨校準 + M67-D OOS 稅）報告（2026-07-21）

工具 `m67e_rf48.py`（永不 ship）｜快取 `m67e_cache.pkl`｜產物 `results_M67E_rf48.json`
既有檔唯一修改 = `rf_score_model.py`（cores 網格加 48 + 48c 結構表，asserts 未動）
**cache-only 純分析；送件形零改動。**

---

## 0. 判定摘要

| 問題 | 答案 |
|---|---|
| 48c 下 `wall = max_i` 假設成立？ | **成立，100/100 案**（`Σ/48` 僅 max 的 3-27%；crossover `c*` 逐帶 max 9.1-27.2 → 連 24 physical 都撐得住） |
| 48c 下 tier-4 / M50 lowcore fail-open？ | **是，100/100 案**（`_effective_cores()=48 > _M45_CORES_MAX=8` → 池與 `_band_env` 逐案 = 12c 版；@4c 對照 40 池/40 案會變 → 閘活的） |
| 送件形（`ADAPTIVE_POOL=1`）在 48c 仍最佳？ | **是，而且差距巨大**：`POOL=0` 全池 **+31~+60%**（即使已給它 +2.75% 的 OOS 品質優勢） |
| M49/M50 REFINE band cut 在 48c 仍要留？ | **要**：還原 K=12 **+10~+21%**（大案 wall 1.95→4.89s） |
| M42/M45 的 build-profile pool cut 在 48c 值多少？ | **恰恰 0.00% wall**——48c 下 wall=max-setter，被砍的 22 隻全都比 max-setter 便宜 → **買不到任何 RF，卻要付 OOS 品質稅** |
| ⇒ Final 決策 | 送件形**先不動**；但「48c 高核 → 跳過 M42/M45 池砍」是**目前唯一還在桌上、上界 −2.1% 的分數槓桿**，前置 = M67-F OOS 實測（見 §6） |

---

## 1. 校準修正（本 session 的核心，改寫 M67-D §3 的算術）

M67-D 寫：「alpha 校準 ⇒ `t ≈ 0.30·median` ⇒ 切法淨賺 ~28%、median ≥ 8.2× t_shipped 才翻盤」。
**該推導把 alpha 的逐案 runtime 當成我們現行 shipped 的 runtime。實際不是：**

- alpha 送的是 **M10（commit 8565e38）**，池 = **14 隻廉價 knob profile**（base / WIRE / ANCHOR / LR-TB aspect / FRAME），無 FREE/FC/CA/MIB/OS16 stack。
- alpha 逐案 grader runtime（`cadc1075_results.json`）：**p50 0.673s、大案 2.4-4.3s、總 96.4s**。
- 我們現行 shipped 本機 12c 實測：p50 1.547s、大案 1.8-2.4s。

⇒ 兩者**同量級**（逐案比值 p50 2.43，大案 0.57-1.4）。median 必須**逐案錨到 `t^alpha`**：

| 模型 | 式子 | 解 | 自洽性 |
|---|---|---|---|
| **A（比例）** | `M_i = κ·t_i^alpha` | RF 均勻 = 0.70802 → **κ = 3.161** | 0.708 > floor 0.70 → 無 clamp，自洽 |
| **B（常數）** | `M_i = M` ∀i | **M = 9.43s** | 97/100 案被 clamp 到 floor、3 案在上方 |

（兩者都由 `官方 1.0286 / raw 1.45278763 = 0.70802` 反解；gate0 已驗 raw 由逐案 cost + `e^(n/12)` 權重逐位重算 = json total，|d|=2.2e-16。）

**機速 s（grader / 本機）**：M10 池只有 3 隻在今日 `_PROFILES` 中逐字存活 → 用兩個 bracket 重估
（mapped-3 → s p50 **1.72**；今日全部 knob-only 10 隻 → s p50 **1.49**；兩者都是**上界**，因 M10 binary 早於 M46 加速）。
⇒ 投影對 **s ∈ {1.0, 1.5, 2.0, 2.5}** 全掃，實測 bracket [1.5, 1.7] 落在中間。

## 2. 48c 結構（gate0 + rf_score_model 新表）

```
band          #  |P|   max_i  Σ/48  ΣPT   c*max  binding
(0,40]       20   35    0.25  0.11  0.03   27.2   max x20
(40,60]      20   35    0.81  0.34  0.06   22.5   max x20
(60,100]     40   26    1.90  0.60  0.09   17.9   max x40
(100,110]    10   13    3.43  0.58  0.08    9.1   max x10
(110,inf]    10   13    4.09  0.68  0.09    9.6   max x10
```

- **48 核對重帶零加速**：n>100 在 12c 就已 max-setter-bound（`max_i 4.97 > Σ/12 3.98`）→ 12c 與 48c 同值。48 核只幫 mid/small（case 70: 2.82→2.11、case 30: 1.46→0.87）。
- **M47 之後 proxy 鏈已幾乎免費**：實測 `_proxy_metrics` ≈ 2.5ms/隻（ΣPT 逐帶 0.03-0.09s，遠低於 max 項）；ledger 裡的「2.9s 尾巴」是 pre-M47 數字。

## 3. wall 模型校準（`fit`，對 100 案實測 12c runtime）

`W(pool,cores) = max(max dt, Σdt/cores, ΣPT)`，逐案 `γ_i = meas12_i / W(ship,12)`：

| band | γ p50 | 解讀 |
|---|---|---|
| (0,40] | 1.202 | 無 REFINE overlay，模型 ≈ 實測（小案固定成本占比高） |
| (40,60] | 1.057 | 同上 |
| (60,100] | 0.880 | = M50 的 K=8 |
| (100,110] | 0.496 | = M49 的 K=4（≈ −50%，與 M49 量到的 band wall −48% 吻合） |
| (110,inf] | 0.463 | 同上 |

- **OLS（40 隻 REFINE-free 案）**：`meas = a·W12 + b·|P| + c` → **a = 0.9997、b = 2.45 ms/profile、c ≈ 0**、|resid| p50 18ms → 模型本身幾乎 exact，**每多一隻 profile 的序列成本只有 2.45ms**。
- **full-pool 側用 M67-D pool0 的實測 12c runtime**（20 隻 n>100）：`γ_full p50 = 1.088`（模型在 41-way oversubscription 下只低估 9%）、**實測 full/shipped wall = 6.27×**。
  ⚠️ 這條同時是「restore 是否真免費」最強的反-contention 證據：12 核跑 41 隻並行只讓模型偏 9%，48 核跑 34 隻的擁塞只會更小。

## 4. 投影（模型 A κ=3.16；完整網格見 `results_M67E_rf48.json`）

品質（RF=1.0 加權，audit cost + OOS 稅錨到 M67-D 實測比值 **+2.751%**）：
`shipped 1.35501 / restore 1.35486 / restoreK12 1.32559 / full 1.32485`（未稅 shipped = 1.32574 = 本地 1.3265 級的 fiction）

逐帶 48c runtime 估計（s=1）：

| band | shipped | restore | restoreK12 | full | alpha t | M_i(A) |
|---|---|---|---|---|---|---|
| (0,40] | 0.27 | 0.27 | 0.27 | 0.99 | 0.16 | 0.51 |
| (40,60] | 0.84 | 0.84 | 0.84 | 2.48 | 0.39 | 1.25 |
| (60,100] | 1.72 | 1.72 | 2.06 | 5.41 | 0.93 | 2.93 |
| (100,110] | 1.62 | 1.62 | 3.74 | 9.67 | 1.74 | 5.49 |
| (110,inf] | 1.95 | 1.95 | 4.89 | 11.54 | 2.43 | 7.70 |

投影官方總分（越低越好）：

| s | shipped | restore | restoreK12 | full | rest% | restK12% | full% | 加權 RF(ship) |
|---|---|---|---|---|---|---|---|---|
| 1.00 | **0.9926** | 0.9925 | 1.1424 | 1.5164 | −0.01% | +15.09% | +52.77% | 0.7325 |
| 1.50 | 1.0747 | 1.0746 | 1.2902 | 1.7125 | −0.01% | +20.05% | +59.35% | 0.7931 |
| 2.00 | 1.1668 | 1.1667 | 1.4065 | 1.8669 | −0.01% | +20.54% | +60.00% | 0.8611 |
| 2.50 | 1.2476 | 1.2475 | 1.5038 | 1.9962 | −0.01% | +20.54% | +60.00% | 0.9207 |

κ∈{2.5,4,6} 與模型 B（M=9.43s）**結論方向完全相同**（full +31~+60%、restoreK12 +10~+21%、restore ≈ 0）；
`--tax-all`（OOS 稅擴到全帶）與 `--cores 24` 亦不改變任何符號。

**RF floor 餘裕** `h = 0.3046·M_i / t_i`（h≥1 = 在 floor 上、有 h 倍鬆弛）：
s=1 時只有重帶部分觸底（(100,110] 6/10、(110,inf] 6/10、h p50 1.02/1.29），mid band **0/40 觸底**（加權 RF 0.815）；s≥2 全部離開 floor。
⇒ **我們並不是「深踩 floor 有 2.5 倍安全邊際」**（M67-D 的說法），而是**貼著 floor 邊緣、mid band 甚至完全在 floor 之上** → 48c 下 runtime 仍是活的分數項。

## 5. 🚨 主發現：48c 下 M42/M45 的池砍買不到任何 RF

48c wall = max-setter ⇒ 任何「自身 dt ≤ 當前 max-setter」的被砍 profile，加回來**不動 wall**：

| band | |ship| | |restore| | +隻數 | dW@48c | ΔΣPT | c*(restore) | dW@24c | in-set dQ |
|---|---|---|---|---|---|---|---|---|
| (0,40] | 35 | 37 | +1.2 | **+0.00%** | +0.00s | 27.2 | +0.53% | −0.046% |
| (40,60] | 35 | 38 | +1.9 | **+0.00%** | +0.00s | 23.9 | +0.00% | −0.163% |
| (60,100] | 26 | 37 | +10.7 | **+0.00%** | +0.04s | 24.2 | +0.02% | −0.011% |
| (100,110] | 13 | 33 | +22.2 | **+0.00%** | +0.13s | 22.0 | +0.00% | +0.000% |
| (110,inf] | 13 | 34 | +22.3 | **+0.00%** | +0.20s | 23.4 | +0.00% | −0.015% |

- 被排除在 restore 之外的正是 **OS16/OM8 那幾隻巨獸**（11-12s ≫ max-setter）→ **M41 的 swap 砍法在 48c 依然是主要 RF 來源**（`full` 欄 +53~60% 就是它 + REFINE）。
- **in-set dQ 恆 ≈ 0 是設計使然**（M42/M45 的閘就是「在這 100 案上逐案 cost 相等才砍」）→ 樣本內完全看不出差別；真正的代價由 M67-D 在 OOS 上量到（**+2.825%**，本模型錨成 +2.751%）。
- 因為 restore 的 runtime 代價 ≈ 0，**break-even θ\* = 0.000**（θ = 回收 OOS 差距的比例）：**只要 OOS 上回收到任何一點，restore 就贏**；θ=1（池砍是 OOS 稅的全部來源）時 **上界 −2.11%**。

⇒ 一句話：**M42/M45 是為 12 核的 `Σ/cores` 牆設計的；Beta 的 48 核把那面牆拿掉了，只剩下它的 OOS 品質稅。**

## 6. 對 Final 的交接（M67-F 規格，本 session 未做）

**不動送件形**——restore 的價值完全取決於「+2.825% 的 OOS 稅有多少來自池砍、多少來自 REFINE band」，這是 cache 無法回答的（M67-D 只量了 shipped 與 POOL=0 兩點；in-set 分解為 REFINE +0.055%、池砍 ≈ +0.05%，各半，但不可外推）。

M67-F（實測，~20-40 分）：
1. `m67_oos_probe.py` 加一個「高核 restore 池」env 形（保留 M41 swap 砍 + M49/M50 REFINE，跳過 `_BIG_REDUNDANT_IDX` 與 `_M45_BAND_DROP`），在 **同一批 80 OOS 重案**上重跑 `pool0` 協定 → 得 θ 的實測值。
2. θ 顯著 > 0 才進 ship 討論；ship 形應鏡射 M45 doctrine 做成 **cores-gated tier-5**（`_effective_cores() >= T` 才跳過池砍，偵測失敗 → 維持現行 = fail-safe），並重跑全鏈：`rf_score_model.py`（常數不需重算，但要重看投影）→ `m49_refine_probe.py` 三 gate → `regression_suite.py` → `make_submission.py all` → M67-C 的 WSL 逐位複驗。
3. ⚠️ 時程：Beta deadline **2026-07-31 17:00 GMT+8**，現行 tar 已四關綠。任何動 wrapper 的改動都要付整條重驗鏈；θ 沒量到之前不要動。

## 7. 誠實範圍

- alpha median 來自 **alpha 場**（pipeline exercise、validation 測資、2026-07-14、我們送 M10）。Beta 場的 median 會不同（對手也會變快）→ 全部結論以 κ×s 網格呈現；符號在 κ∈[2.5,6]、s∈[1,2.5]、模型 A/B 全部一致。
- s 的兩個估計都是**上界**（含 M10 binary vs 今日 binary 的速度差）；s 偏大 = 對 shipped 較不利 = 保守。
- audit dt 是 K=12 counterfactual；本報告用「實測 12c runtime × 模型比值」規避絕對偏差，`fit` 的 a=0.9997 是該做法的驗證。
- OOS 稅只在 **n>100（81% 權重）** 量過，且未分離 pool cut 與 REFINE；`--tax-all` 是保守敏感度。
- 48c 假設 48 個可並行執行緒。若實為 24 physical + HT：`c*(restore) ≤ 27.2`、`dW@24c ≤ +0.53%`（只在最輕的 (0,40] 帶）→ 結論不變。
- restore 的「wall 不變」未在真 48 核機上實測（本機只有 12 核）；最強的間接證據 = 12 核跑 41-way 並行時模型只低估 9%（§3）。M67-F 若做，應順帶在多核機（WSL2/GPU box）計時。

## 8. 重現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py m67e_rf48.py gate0      # 五閘（含 rf_score_model 子行程；~5 分）
& $py m67e_rf48.py calib      # κ / 常數 M / 機速 bracket
& $py m67e_rf48.py fit        # wall 模型校準 + per-profile overhead OLS
& $py m67e_rf48.py project    # 投影網格 + floor 餘裕 + free-restore 預算
& $py m67e_rf48.py report     # 上述 + results_M67E_rf48.json
# 敏感度： --tax-all / --cores 24 / --speeds 1 3 / --slack 1.2
```
存證：`m67e_gate0_stdout.txt`（五閘）、`m67e_rfmodel_stdout.txt`（rf_score_model 全綠 + 48c 表）、`m67e_report_stdout.txt`。
