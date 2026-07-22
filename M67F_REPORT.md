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
| θ_mid（(60,100] 的 M45 tier-3；**§7 補量**） | **0.5913**（品質 GREEN）**但 ship 判 RED** —— 加進來每個 s 都變差 |
| ⚠️ §3 那張 −1.60~−1.62% 表 | **高估**：投影用了 dt-filtered 池、θ 卻是 index 池量的。正解 **−0.26 ~ −1.30%**，見 §7-2 |
| ⚠️ M45 tier-3 strict gate | 在**出貨組態**（K=8）下**已失效**（in-set case 64 +0.41%，加權 −0.0018%），見 §7-4 |

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

1. ~~**θ 只覆蓋 n>100**~~ → **已補齊，見 §7**（mid band θ_mid = 0.5913，但 **ship 判 RED**）。
2. ~~**index-based vs dt-based 等價**~~ → **這句話是錯的，已由 §7-2 更正**：兩者在 n>100 的
   *隻數* 相近（22 vs 22.3）但 *身分* 不同，index 版會還原真正貴的那幾隻 → **48c wall +5.7~8.7%**，
   不是 wall-free。§3 的 −1.60~−1.62% 因此是**高估**，正確值見 §7-2。
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
  才跳過 **M42（只有 M42）**；偵測失敗 / 未知 → **現行行為**。
  ⚠️ **tier-3 不要一起放**——§7 已補量並判 RED（加進來每個 s 都變差 0.01~0.02pp）。
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

---

# §7 mid band 補量（2026-07-22 同日追加）——θ_mid 綠、**ship 判 RED**；附兩項更正

工具改動（皆 offline，**`optimizer_constructive.py` 一字未動、tar 未重打包**）：
`m67_oos_probe.py` 加 `--pool0-hi`（`pool0`/`restore` 的上界，配 `--pool0-lo` 就能單獨計分一個帶）
+ `_theta_report` 的**逐帶分解**；`m67e_rf48.py` 加 `pool_restore_idx()` / `restoreIdx` 變體
+ `--idx-bands` + mid 帶稅錨與 `--theta-mid`。

## 7-1 θ_mid = 0.5913（品質面 GREEN）

`pool0 --pool0-lo 60`（新解 40 in-set + 80 OOS mid 案）→ `restore --arm pool --pool0-lo 60`。
同一批抽樣、同估計量，重案端點全部沿用 cache（heavy 那 80 案零重跑，數字逐位重現 Phase 1）。

| band | wContr | shipped | restore | full | 分母 S/F−1 | restore 回收 | **θ** | movers |
|---|---|---|---|---|---|---|---|---|
| **(60,100]** | 18.2% | 1.618048 | 1.608076 | 1.601185 | **+1.053%** | +0.620% | **0.5913** | 21/80（20 better / 1 worse）|
| (100,130] | 81.1% | 1.659884 | 1.625062 | 1.614282 | +2.825% | +2.143% | 0.7636 | 45/80（44 / 1）|
| 合併 n>60 | 99.3% | 1.652213 | 1.621947 | 1.611881 | +2.502% | +1.866% | 0.7504 | 66/160（64 / 2）|

- **mid 帶的 OOS 稅只有 heavy 的 37%**（+1.053% vs +2.825%）——tier-3 砍 9 隻、M42 砍 22 隻，
  砍得少、稅也少；方向一致（20 better / 1 worse）。
- heavy 列**逐位重現 Phase 1**（1.659884 / 1.625062 / 1.614282、θ=0.7636、44/1）= 本次改動沒污染舊結果。
- 12 核 wall（n>60）：shipped 1.84s → restore 2.72s → full 8.79s。

## 7-2 🚨 更正：Phase 1 的 −1.60% 是用**錯的池**投影的

θ 是拿 `ICCAD_M67F_RESTORE=1` 量的 = **index-based**（把 drop set 的 22/9 隻**全部**放回來）。
但 §3 投影呼叫的是 `m67e_rf48.py` 的 `restore` 變體 = **dt-based**（`restore_candidates()` 只撿
`dt ≤ max-setter` 的），**wall-free 是它的定義、不是它的發現**。兩者在 n>100 隻數相近（22 vs 22.3）
但**身分不同**：index 版會放回 #5/#15/#16/#18 這些在某些案上**本身就是 max-setter** 的 profile。

新變體 `restoreIdx`（對 100/100 案與 knob 逐案**池身分完全相同**，已 assert）：

| band | dW@48c（index restore） | movers | dt-filtered 變體 |
|---|---|---|---|
| (0,40] / (40,60] | +0.00% | 0/20, 0/20 | +0.00%（定義使然）|
| (60,100] | **+2.34%** | 9/40 | +0.00% |
| (100,110] | **+8.70%** | 5/10 | +0.00% |
| (110,inf] | **+5.68%** | 7/10 | +0.00% |

代進模型（κ=3.161、48c、θ_big=0.7636、θ_mid=0.5913；mid 帶稅現在也加在 shipped 側 ⇒ 基準
0.9926→0.9947）：

| s | shipped | restoreIdx（**只放 M42**） | Δ | 〔對照〕dt-filtered |
|---|---|---|---|---|
| 1.00 | 0.9947 | 0.9817 | **−1.30%** | −1.74% |
| 1.50 | 1.0771 | 1.0712 | **−0.55%** | −1.73% |
| 2.00 | 1.1694 | 1.1664 | **−0.26%** | −1.73% |
| 2.50 | 1.2504 | 1.2471 | **−0.26%** | −1.73% |

**⇒ 真實可 ship 的上檔是 −0.26 ~ −1.30%（隨機速 s 遞減），不是 −1.60~−1.62%。** 符號仍然對，
但量級掉一半以上，而且**最樂觀的那格（s=1）才有 −1.3%**。Phase 2 的多核 wall 實測因此更關鍵：
模型說 index restore 在 48c 要付 +5.7~8.7% wall，這個數字必須用真機驗。

## 7-3 mid band ship 判定 = **RED**（品質綠但淨值負）

| `--idx-bands` | s=1.00 | s=1.50 | s=2.00 | s=2.50 |
|---|---|---|---|---|
| `big`（只放 M42） | **−1.30%** | **−0.55%** | **−0.26%** | **−0.26%** |
| `mid,big`（加放 tier-3） | −1.29% | −0.53% | −0.24% | −0.24% |

**加上 mid 帶在每一個 s 都比較差**（+0.01~0.02pp）。算術一行：mid 帶 restore 的品質回收
**+0.620%**，wall 代價 `(1+0.0234)^0.3 = ` **+0.695%** → 淨負；而且 M67-E 已證 mid band
**0/40 觸底**（完全在 RF floor 之上）⇒ 那 +2.34% wall **全額付**、沒有 floor 吸收。
⇒ **Phase 3 的 tier-5 只放 M42，tier-3 維持現狀。** mid 帶 break-even 需要 θ_mid ≥ 0.65 左右
（實測 0.5913，差一截）。

## 7-4 🚨 附帶發現：M45 tier-3 的 strict gate 在**出貨組態**下已失效

Gate B 把窗口拉到 n>60 後（同時檢 M42 與 tier-3），60 案裡 **1 案不相等**：

```
case 64 (n=85)   shipped 1.3580617332260372   restore 1.3524850027999449   -0.4106%
```

單案複驗（`shipped` / `restore` × `K=8` / `K=12`，鄰居 63/65 當對照全部四格恆等）：

| 組態 | shipped 池(26) | restore 池(35) | 相等？ |
|---|---|---|---|
| REFINE **K=12**（= 當年推導 tier-3 的組態）| 1.3558352796522921 | 1.3558352796522921 | ✅ |
| REFINE **K=8**（= **實際出貨**，M50）| 1.3580617332260372 | 1.3524850027999449 | ❌ **+0.41%** |

**機制**：`_M45_BAND_DROP`（與 `_BIG_REDUNDANT_IDX`）是從 `audit_cache.pkl` 推的，那是
**REFINE=12** 的 positions；M49/M50 之後在**同樣那兩個帶**疊了 K=4/K=8 overlay ⇒
**strict selection-preserving gate 從來沒有在出貨組態下重證過**。heavy 帶運氣好（20/20 仍相等），
mid 帶破在 case 64。

- **對送件的影響：可忽略**——case 64 權重僅 0.433%，加權後 **−0.0018%**（local total 會從
  1.326473104916827 變成 1.326448970）。方向是「我們現在比可達的略差」，不是正確性問題。
- **對方法論的影響：不可忽略**——ledger 那句「M42/M45 逐案 cost 相等 ⇒ local 逐位不變」
  對 tier-3 **已不再嚴格成立**。任何**重算 tier-3/M42 常數**的動作都必須在 K=4/K=8 overlay 下做，
  不能再直接吃 `audit_cache.pkl`。
- Gate B 因此改成兩段：**restore 比 shipped 差 = 硬 FAIL**（proxy 樣本內是 oracle-min，
  超集池只可能弱贏 ⇒ 變差就是 knob 壞了）；**restore 比 shipped 好 = DRIFT，印出來但繼續**
  （這是關於出貨常數的發現，不是本量測的故障）。

## 7-5 重現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py m67_oos_probe.py pool0   --pool0-lo 60                 # 40 in-set + 80 OOS mid（~13 分）
& $py m67_oos_probe.py restore --arm pool --pool0-lo 60      # Gate A/B + 80 案 + 逐帶分解（~5 分）
& $py m67e_rf48.py project --skip-rfmodel --idx-bands big      # ship 候選形
& $py m67e_rf48.py project --skip-rfmodel --idx-bands mid,big  # 加 tier-3 = 更差
# 只要 mid 一帶：--pool0-lo 60 --pool0-hi 100（另存 results_M67F_theta_pool_60_100.json）
```

存證：`m67f_mid_pool0_stdout.txt`、`m67f_mid_restore_stdout.txt`、`m67f_mid_proj_big.txt`、
`m67f_mid_proj_midbig.txt`、`results_M67F_theta_pool_60_inf.json`。
