# M67-F Phase 1 — θ 實測（+2.825% OOS 稅裡屬於 M42/M45 池砍的比例）報告（2026-07-22）

工具 `m67_oos_probe.py` 新 mode `restore`（永不 ship）｜快取 `m67_oos_cache.pkl` 桶 `m67f`
產物 `results_M67F_theta_pool.json` / `results_M67F_theta_refine.json`
既有檔唯一修改 = `optimizer_constructive.py` 加 offline 旋鈕 `ICCAD_M67F_RESTORE`（**預設 off ⇒ 送件行為逐位不變**）
**送件形零改動、tar 未重打包**（`b9589618d507de0561f79a55a80fd8f3` 仍是四關綠的那份）。

---

## 0. 判定摘要

| 問題 | 答案 |
|---|---|
| θ_pool（M42/M45 池砍佔 OOS 稅的比例） | **0.7636**（80 案）／0.6745（20 案 pilot 檢查點） → **GREEN（≥0.30）** |
| θ_refine（M49/M50 REFINE band 那層） | **0.0864** → RED；且雙向（15 better / 6 worse）= OOS 上近乎品質中性 |
| 兩者相加 | 0.8500（vs 1.0 → 15% 是**交互作用**，方向 = 兩層一起還原比分項和多回收） |
| 在 M67-E 模型下值多少官方分 | **−1.60 ~ −1.62%**（s∈[1,2.5] 全同號；`--tax-all` 保守版 −2.06%；θ=1 上界 −2.11%） |
| ⇒ Phase 1 結論 | **GREEN，進 Phase 2**（真多核機 wall 實測）。**本 session 不動送件形**。 |

一句話：**M67-E 說「M42/M45 在 48 核買不到 RF」；M67-F 量到它們同時是 OOS 品質稅的主要來源（76%）。
兩件事合起來 = 目前桌上唯一一個上界 −2% 級、且方向確定的分數槓桿。**

---

## 1. 量測設計（與 M67-D `pool0` 完全同協定）

四個點，同一批 80 案 OOS 重案（訓練語料 `floorset_lite`，n>100，per-n 先平均再官方加權）：

| 變體 | env | 池（n>100） | REFINE |
|---|---|---|---|
| `shipped` | 預設 | 13 | K=4 |
| `restore`（新） | `ICCAD_M67F_RESTORE=1` | **35**（+22 = `_BIG_REDUNDANT_IDX`） | K=4（保留） |
| `norefine`（第三臂） | `ICCAD_ADAPTIVE_REFINE=0` | 13（不動） | K=12 |
| `full` | `ICCAD_ADAPTIVE_POOL=0` | 41 | K=12 |

`θ_arm = (S − R_arm) / (S − F)`。shipped 與 full 兩端點直接沿用 M67-D 的 cache（`_sig` 只含
`_PROFILES`+exe md5，旋鈕不動 `_PROFILES` ⇒ 240 案 shipped + 80 案 full 全數重用，零重跑）。

**n>100 只有 M42 層在動**：tier-3 `_M45_BAND_DROP` 是 (60,100]、tier-4 需 `cores≤8`（12c/48c 皆
fail-open）⇒ 本次 θ 嚴格說是 **M42 層**的 θ。

### 閘

- **Gate A（旋鈕自檢，先跑不解案）**：knob off 逐帶池 = 35/35/26/13/13 且 REFINE band = mid 8 / big 4；
  knob on = 35/35/35/35/35 且 **REFINE band 不變**；`restore(120) − shipped(120)` **恰等於**
  `_BIG_REDUNDANT_IDX`。arm=refine 對照：池不動、band 清空。兩 arm 五閘全 PASS。
- **Gate B（in-set strict-gate 複驗）**：restore 池在 100 案 validation 的 20 個 n>100 案上
  逐案 cost 對 `results_shipped_m51.json` → **20/20 相等（rel 1e-9）**。
  這同時複驗了 M42 當年的 strict selection-preserving 閘至今未漂移。
- **Off-path 純淨性（改 wrapper 後立刻做）**：
  - V1 `_pool_indices(n)`/`_band_env(n)` 對 n=1..130 與 `git show HEAD` 版**逐一相等**；
  - V2 官方 eval total **1.326473104916827 逐位**、`m53_diff_results` **0 movers**、逐案 positions
    亦逐位相同、100/100 feasible（`results_M67F_offpath.json`）；
  - V3 `m48_coldstart_dryrun.py` 四 phase + `opwrapper` 四 phase 全 PASS。

---

## 2. 結果

```
theta_20 (pilot, 檢查點)  shipped 1.686364  restore 1.652386  full 1.635990
                          分母 +3.079%   restore 回收 +2.056%   theta = 0.6745
                          movers 10/20（10 better, 0 worse）

theta_80 (判定樣本)       shipped 1.659884  restore 1.625062  full 1.614282
                          分母 +2.825%   restore 回收 +2.143%   theta = 0.7636
                          movers 45/80（44 better, 1 worse）
```

- **方向極其一致**：45 個 mover 裡 **44 個變好**，唯一退步案 +0.01%（雜訊級）。
  改善幅度 mean 3.34% / median 3.07% / max **10.65%**。
- **restore 已經吃掉 full 的大部分好處**：48/80 案 restore 的 cost **等於** full（proxy 在 35 隻池裡
  已經選到 full 會選的那隻），另有 **9 案 restore 比 full 更好**（full 額外還原的 swap 巨獸 + K=12
  反而把 proxy 帶偏）。
- **θ_refine = 0.0864（RED）**：M49/M50 的 REFINE band cut 在 OOS 上幾乎不欠品質債
  （15 better / 6 worse、淨 +0.238%），與 M49/M50 當初「local +0.027%/+0.028%」的 in-set 結論一致。
- **交互作用**：0.7636 + 0.0864 = 0.8500 ≠ 1 ⇒ 約 15% 的 OOS 稅只有在**兩層同時還原**時才回收
  （super-additive）。這不影響決策（θ_pool 單獨就 0.76），但報告必須註明：
  「restore 免費」的解讀是**針對池砍那一層**，不是說 REFINE 也該還原——後者在 48c 是 **+15~21% 的
  RF 主力**（M67-E §4），且 OOS 品質債只有 8.6%，**必須留**。

### 12 核 wall（**不可外推 48c**）

| 變體 | 均值 | vs shipped |
|---|---|---|
| shipped | 1.59s | 1.00× |
| restore | 3.17s | **1.99×** |
| norefine | 3.22s | 2.02× |
| full | 11.59s | 7.28× |

本機是 `Σ/cores`-bound（12 核跑 35 隻），Beta 是 max-setter-bound（48 核跑 35 隻，M67-E 證 100/100
案 `Σ/48` 只有 max 的 3-27%）⇒ **這個 1.99× 不是 48c 的預期值**，48c 的預期是 +0.00%（模型）。
Phase 2 就是去把這句話變成實測。

---

## 3. 分數投影（M67-E 模型，把 θ 代入 `variant_quality(theta)`）

模型 A、κ=3.161、48 核、OOS 稅只加在 n>100：

| s（機速） | shipped | restore @θ=0.7636 | restore @θ=1 | Δ | 上界 |
|---|---|---|---|---|---|
| 1.00 | 0.9926 | **0.9765** | 0.9716 | **−1.62%** | −2.11% |
| 1.50 | 1.0747 | 1.0575 | 1.0523 | −1.60% | −2.09% |
| 2.00 | 1.1668 | 1.1481 | 1.1425 | −1.60% | −2.09% |
| 2.50 | 1.2476 | 1.2276 | 1.2216 | −1.60% | −2.09% |

`--tax-all`（稅擴到全帶，保守）：**−2.06%**（上界 −2.69%）。**符號在整個 s 網格上不變。**

重現：`m67e_rf48.py project --skip-rfmodel`（M67-F 給該工具加了 `--theta`，**預設就是 0.7636**；
`--theta -1` 可關掉這段）。

---

## 4. 誠實範圍（重要，勿在後續 session 忘記）

1. **θ 只覆蓋 n>100（81% wContr）= M42 層**。`(60,100]` 的 M45 tier-3（9 隻、18.2% wContr）
   **OOS 稅未量**——cache 沒有 mid band 的 full-pool 對照，補量需要 mid 帶的 `pool0` + `restore`
   兩批新 run（各約 80 案）。Phase 3 若要把 tier-3 也一起鬆綁，**必須先補這一批**。
2. **本 knob 的 restore 是 index-based**（還原兩個 drop set 的全部 22/9 隻）；M67-E 模型的 restore 是
   **dt-based**（`dt ≤ max-setter` 才撿）。n>100 兩者是 22 vs 22.3 隻 → 等價；小案帶（M67-E 會多撿
   幾隻便宜 swap profile）不在本量測內。
3. **12 核 wall 不能推 48c**（見 §2）。「48c 免費」目前仍只有 M67-E 的模型 + 間接證據
   （12 核跑 41-way 並行時 wall 模型只低估 9%）⇒ **Phase 2 是 ship 的必要條件**，不是加分項。
4. 投影用的 alpha median 錨（κ）來自 **alpha 場**；Beta 場 median 會不同 → 全網格呈現，符號不變。
5. OOS 語料 = 訓練集 `floorset_lite`，M67-D 已證它**比測試語料更硬**（label floor 1.2444 vs 1.1079）。
   θ 是**比值**，對語料難度的一階效應會抵銷，但沒有嚴格證明它與難度無關。

---

## 5. 交接：Phase 2 / Phase 3

**Phase 2（下一步，必要條件）**：真多核機實測 restore 池的逐案 wall。
- 機器 = GPU box 的 WSL2（先確認 `nproc`；本機只有 12 核，結構上答不了這題）。
- 判準（**預註冊**）：restore 相對 shipped 的逐案 wall 漲幅 **>2% ⇒ 免費論證破產、結案**；
  ≤2% ⇒ 進 Phase 3。
- 只要量 wall，不需要重跑品質（品質已由本 session 的 θ 定案）。

**Phase 3（θ 綠 ∧ 多核 wall 綠才開）**：
- ship 形 = **cores-gated tier-5**，鏡射 M45 doctrine：`_effective_cores() >= T`（建議 T=32）
  才跳過 M42（tier-3 要不要一起放，取決於 §4-1 的 mid band 補量）；偵測失敗 / 未知 → **現行行為**。
- 全鏈重驗一項不能省：`rf_score_model.py` 投影 → `m49_refine_probe.py` `variant 4 big`/`8 mid`/`4 mid`
  → `regression_suite.py` 六項 → `make_submission.py all` → M67-C 的 WSL `verify_final_tar.sh` 逐位。
- ⚠️ deadline **2026-07-31 17:00 GMT+8**。現行 tar 已四關綠 → **先上傳現行 tar**；換件前必須先確認
  Drive 可否覆蓋重傳。時間不夠 → 維持現行送件形，把 M67-F 結論留給 Final。

---

## 6. 重現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py m67_oos_probe.py restore --arm pool      # Gate A + Gate B + 80 案 + theta（~9 分）
& $py m67_oos_probe.py restore --arm refine    # 第三臂（~6 分）
# off-path 純淨性
cd iccad2026contest; & $py iccad2026_evaluate.py --evaluate ../optimizer_constructive.py `
    --output ../results_M67F_offpath.json; cd ..
& $py m53_diff_results.py results_shipped_m51.json results_M67F_offpath.json 1e-12   # 0 movers
& $py m48_coldstart_dryrun.py; & $py m48_coldstart_dryrun.py opwrapper
```

存證：`m67f_pool_stdout.txt`、`m67f_refine_stdout.txt`、`m67f_eval_stdout.txt`、`m67f_m48_stdout.txt`。
