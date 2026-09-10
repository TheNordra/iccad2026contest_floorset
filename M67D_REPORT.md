# M67-D — OOS 泛化預檢報告（2026-07-21）

工具 `m67_oos_probe.py`（永不 ship）｜快取 `m67_oos_cache.pkl`｜產物 `results_M67D_oos.json`
語料 = `floorset_lite` 訓練集（30240 layouts，從未參與任何 M1-M51 調參）｜seed 67｜240 案

---

## 0. 判定摘要

| 指標 | 值 | 預註冊 bar | 判定 |
|---|---|---|---|
| OOS raw 加權總分 | **1.6533** | ≤1.40 綠 / >1.45 紅 | **RED** |
| floor-relative ratio（OOS/floor） | **1.3287** vs in-set 1.1972 | \|Δ\|≤3% | **+10.98% → CHECK** |
| 硬旗標（infeasible / fallback / 例外） | **0 / 0 / 0** | 任一觸發即紅 | **全清** |
| runtime p50 / p90 / max | 1.54 / 2.44 / 4.67s（in-set 1.55 / 2.58 / 3.56s） | — | **無差異** |

**但 raw 的 RED 不是過擬合證據**——兩條歸因測試證明 gap 是語料難度，不是調參：

1. **單 profile 參照**：single base profile 自己就有 **+26.4%** 的 OOS gap（1.4775→1.8681），比 portfolio 的 +24.6% **更大**；portfolio 增益 in-set 10.22% vs **OOS 11.50%** → 品質軸（M29-M51 free-aspect 家族、proxy `_RH`、池組成）**泛化，甚至 OOS 賺更多**。
2. **語料本身更硬**：label floor 1.2444 vs 1.1079（label 自身 vrel 0.1081 vs 0.0504）、boundary blocks 28.8 vs 24.0（max 46 vs 37）、preplaced 3.08 vs 2.59（max 10 vs 9）、b2b edges 1201 vs 994。

⇒ **真正的發現不在 raw，而在下面第 3 節：adaptive 切法的 OOS 品質稅是 in-set 的 27 倍。**

---

## 1. 方法與閘

- **抽樣**：validation 恰好每個 `n∈[21,120]` 1 案 → 鏡射之，每 n 抽 K 案（K=2；`n>100` K=4 降重權區噪聲）= **240 案 / 100 個 n**。seeded（67）優先取不同檔；訓練單檔 112 layouts 且 n 固定。
- **baseline**：逐位鏡射 `ContestEvaluator._extract_baseline`（`iccad2026_evaluate.py:806`）含 stored-metrics 分支；訓練 `metrics_sol[0]/[-2]/[-1]` 與由 fp_sol 重算相符（max rel dev 2.8e-8）。
- **評分**：官方 `evaluate_solution` + `target_positions`（硬約束全開）+ `median_runtime=1.0`（**RF=1.0**，與本地 1.3265 同語意）。
- **估計量**：per-n 先平均、再套官方加權 → K 不等亦與 validation 權重形一致（naive 全案版 1.6563，差 0.3%，一致性 OK）。
- **gate0 全 PASS**：env 零 `ICCAD_*`、池=41、in-set 3 案（0/50/99）**cost 與 positions 逐位** = `results_shipped_m51.json`、訓練 5 案 fp_sol verbatim feasible ∧ \|hgap\|<1e-8 ∧ agap=0。

## 2. 分帶與最壞案

| band | #n | wContr | OOS mean | in-set | delta | OOS ratio |
|---|---|---|---|---|---|---|
| S (20,60] | 40 | 0.6% | 1.8239 | 1.3727 | +0.4512 | 1.3214 |
| M (60,100] | 40 | 18.2% | 1.6180 | 1.3888 | +0.2293 | 1.2809 |
| **B (100,130]** | 20 | **81.1%** | 1.6599 | 1.3121 | +0.3478 | 1.3397 |

逐 n 配對（validation 每 n 恰 1 案）最大貢獻：n=118 +0.554、120 +0.409、115 +0.588、119 +0.361、113 +0.568。
最壞單案（加權）`worker_7/layouts_5824/L73` n=118 cost 2.0215（floor 1.2801、R=1.58、vb/vg/vm = 8/4/6）。
品質分解（加權）：hgap 0.3230 vs 0.2683、agap 0.2551 vs 0.1837、**vrel 0.1225 vs 0.0384**——其中 label 自帶 0.1081 vs 0.0504，即我們在 OOS 只比 label 多 +0.014 vrel，在 in-set 反而比 label **少** 0.012。

## 3. 🚨 主發現：adaptive 切法的 OOS 品質稅 = in-set 的 27 倍

M41/M42/M45（pool cuts）+ M49/M50（REFINE band）全部是在**這 100 案**上以 "strict selection-preserving"（逐案 cost 相等才砍）推導的。把 `ICCAD_ADAPTIVE_POOL=0`（全 41 池 + full REFINE）在兩個語料的重權帶重跑：

| n>100 | shipped | full-pool | 品質稅 | movers |
|---|---|---|---|---|
| in-set（20 案） | 1.312108 | 1.310721 | **+0.106%** | 3/20 |
| **OOS（80 案）** | 1.659884 | 1.614282 | **+2.825%** | **52/80**（shipped 較差 51 案） |

最大單案退步：+11.92%（n=118）、+11.67%（n=120）、+10.87%（n=107）。

**這正是 M55 五折 CV 預言的事，且更嚴重**：M55 說 strict preservation OOS break rate 40-48%、最壞單案 +9.6%、worst-cell tax +2.36%；真 OOS 實測 **break 65%（52/80）、最壞單案 +11.9%、tax +2.83%**。⇒ 「strict gate ⇒ ∀median ∀cores 弱贏」只在**樣本內**成立，hidden set 上是真實的品質交易。

### 但送件形不動——RF 側算術仍以 10× 差距勝出

- wall 比值：本機 12c 實測 shipped 1.59s vs full-pool 11.59s = **7.28×**（含 REFINE band）；扣掉 REFINE 只算 pool cut（`audit_cache`，兩側同 K=12）中位 **2.57× @12c / 2.50× @48c**。取最保守的 2.5×：
- alpha 校準（`[[alpha-results-2026-07]]`）：我們 cost-加權 RF = 0.7081 ≈ floor ⇒ `t ≈ 0.30·median`。
- 則 full-pool `R = 0.30×2.5 = 0.76 → RF = 0.921`，shipped 仍在 floor 0.70：
  `cost_full/cost_ship = (1/1.02825) × (0.921/0.700) = **1.279`** → **切法淨賺 ~28%**（若 wall 比值取含 REFINE 的 5×，淨賺 ~58%）。
- 切法變成淨虧的唯一條件：**兩側都撞 floor**，即 `median ≥ t_full/0.3046 ≈ 8.2× t_shipped`。alpha 實測 median ≈ **3.28×** → 距離該 regime 還有 2.5 倍安全邊際。

⇒ **不改送件形**。但 M67-E（48c RF 投影）**必須改用 +2.8% 這個 OOS 稅**，而不是 in-sample 的 +0.1%。

## 4. 誠實範圍

- 訓練語料 ≠ Beta hidden set。alpha 已證 alpha 測資 = validation 逐位相同（測試集生成器），訓練集的 label vrel 高一倍、boundary/preplaced 更多，**Beta 若沿用測試集生成器，raw 應該接近 1.33 而非 1.65**；本測給的是「不同生成器下的下界式壓力測試」。
- pool0 只跑 `n>100`（81% 權重），中小帶未測；`ICCAD_ADAPTIVE_POOL=0` 同時關 pool cut 與 REFINE band，兩者未分離（分離估計靠 audit_cache）。
- audit_cache 的 dt 是 n>100 的 K=12 counterfactual（M49 註記），故上面 2.5×/2.57× 是**低估** shipped 側速度優勢的保守值。
- 48c 的 wall 模型 = `max(max_i t_i, Σt/cores)`，未含 wrapper 開銷與 grader 機速差異。
- ref 的單 profile = `ICCAD_CONSTRUCTIVE_SINGLE=1`（`free_aspect` 基礎 profile），非 M1 的空 base。

## 5. 重現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py m67_oos_probe.py gate0     # 四項閘（含 in-set 逐位）
& $py m67_oos_probe.py run       # 240 案，424s
& $py m67_oos_probe.py report    # 報表 + results_M67D_oos.json
& $py m67_oos_probe.py ref       # 單 profile 歸因（340 案，~6 分）
& $py m67_oos_probe.py pool0     # adaptive 切法 OOS 稅（100 案，~20 分）
```
快取斷點續跑；`sig` 只綁 exe md5 + `_PROFILES`（抽樣旗標不入 sig，加大 K 可沿用既有結果）。
