# M80 — 隨機聯合抽樣的旋鈕向量，做成 cores-gated 出貨 tier

**2026-08-05。判定 GREEN，已進 tree（`ICCAD_M80_TIER`，≥40 核才開）。**

## 一句話

M79 的副產物不是雜訊：把旋鈕空間**隨機聯合抽樣**得到的 8 隻固定 profile 掛成
高核 tier，**兩份 disjoint 的 OOS 240 案樣本上 NET@48c 分別是 +1.786% 與 +1.909%**
（bar 0.30%）。這是自 M71 以來最大的單一 lever，而且它推翻的是一個寫了 30 個
milestone 的結論——「古典旋鈕線已經收斂」。

## 為什麼這條會被漏掉 30 個 milestone

M30/M31 掃這個空間的方式是**逐 knob、從人工堆疊的 recipe 往外走、低於 0.05% 就停**，
掃到飽和時最好的單一新 profile 是 **≤0.063%**。

隨機**聯合**抽樣的單一最佳向量是 **+0.439%**，是它的 **7×**。看被挑中的 `#100` 就知道
為什麼座標式貪婪到不了：它同時把 `BP_WEIGHT` 拉到 274048、`MIB_ASPECT` 推到 tall 側
0.2338、frame scale 放寬到 1.45 —— **這三條在死路 ledger 裡各自都被判過死**
（「BP_WEIGHT 雙向封卷」「MIB tall 側 +0.027% 低於 bar」「FRAME_ASPECTS 封卷」）。

> 🔑 **單獨死不代表聯合死。** 凡是「某某旋鈕封卷」的結論，只對**單軸掃描**成立。

## 數字

### 加大 cloud：R=128 → 256（seed 79，prefix-stable）

`build_cloud()` 在 R 上是 prefix-stable（rng 序列不變 ⇒ 前 128 隻接受順序不變），
實測驗證前 128 隻逐字相同 ⇒ 加大只付新向量的錢（12800 新 runs / 797s）。

| | R=128 | **R=256** |
|---|---:|---:|
| 單一最佳新向量（in-sample） | +0.439% | +0.439%（同一隻 `#100`） |
| K=8 in-sample | +1.576% | **+2.075%** |
| K=8 **5-fold held-out** | +0.791% | **+1.000%** |
| per-case oracle | +2.025% | **+2.649%** |

第 2、3 名的單隻向量換成新抽到的 `#182`（+0.394%）與 `#133`（+0.386%）——
加大 R 的回報是真的，不是雜訊。

### in-sample（100 validation 案）與 5-fold CV

| K | in-sample | 5-fold held-out | dRF@48c | NET@48c |
|---:|---:|---:|---:|---:|
| 1 | +0.439% | 0.131% | +0.041% | +0.398% |
| 4 | +1.362% | 0.539% | +0.118% | +1.244% |
| **8** | **+2.075%** | **1.000%** | **+0.241%** | **+1.834%** |
| 11 | +2.303% | **1.304%**（峰值） | +0.259% | +2.044% |
| 12 | +2.368% | 1.293% | +0.343% | +2.025% |

### OOS 240 案 ×2 份 disjoint 樣本 @48c（**判定用的數字**）

錨：s1 shipped `1.555854672`（與 M76 的 `1.555855` 相符）、s2 shipped `1.557813659`。

| K | s1 quality | s1 dRF | **s1 NET** | s2 quality | s2 dRF | **s2 NET** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.299% | +0.066% | +0.233% | +0.101% | +0.000% | +0.101% |
| 2 | +1.219% | +0.249% | +0.970% | +1.096% | +0.000% | +1.096% |
| 4 | +1.626% | +0.263% | +1.363% | +1.465% | +0.007% | +1.458% |
| **8** | **+2.073%** | **+0.287%** | **+1.786%** | **+1.920%** | **+0.011%** | **+1.909%** |
| 12 | +2.092% | +0.287% | +1.805% | +1.934% | +0.011% | +1.923% |

**bar = NET 0.30%**（M75/M76/M78 的 pre-registered OOS bar）⇒ **過 bar 約 6 倍**。

帶別分解（K=12，s1 / s2 的 quality）：
`(20,60]` +1.424% / +0.888%、`(60,100]` +2.512% / +2.380%、`(100,130]` +2.005% / +1.838%
⇒ **三個帶都是正的**，重帶（權重 53%）拿到 ~1.9-2.0%。

### 為什麼 K=8

兩份樣本在 K=8 都有**乾淨的手肘**：第 8 隻（`#170`）值 +0.195pp（s1）／+0.249pp（s2），
第 9 隻只值 +0.004pp／+0.009pp。K=12 只多買 +0.019pp／+0.014pp，卻讓池大 50%。
池大小就是「評分機**有效**並行度低於偵測值」時的曝險（與 tier-5 同一個賭注）
⇒ 為了 1% 的增益去多付 50% 曝險不划算。**K 是在 OOS 上挑的，不是 in-sample 曲線。**

### 為什麼一定要 cores-gated

同一批向量、同一份 100 案，只換池形狀：

| | quality | dRF | **NET** | 被抬 wall 的案子 |
|---|---:|---:|---:|---:|
| **@48c**（tier-5 ON） | +2.075% | +0.241% | **+1.834%** | 39/100 |
| **@12c**（tier-5 off、tier-3 ON） | +2.075% | **+10.619%** | **−8.544%** | **100/100** |

機制是 wall = `max(max_i dt_i, Σdt_i / cores)`：48 核上 100/100 案是 max-setter-bound，
所以每隻都比現任 max 便宜的候選幾乎免費；12 核上 `Σ/cores` 主導，K=8 直接把每一案的
wall 都抬起來。**這不是微調，是 +1.83% 與 −8.54% 的差別。**
（+10.619% 獨立重現了 M79 手推的 +10.614%。）

## 實作

```python
_M80_EXTRA  = [...8 隻...]           # append 在 _M55_BASE_LEN 之後（idx 86-93）
_M80_IDX    = frozenset(range(86, 94))
_M80_CORES_MIN = 40                  # 與 _M67F_CORES_MIN 同值、獨立可調
```

- `_m80_active(n)`：**call time** 讀 env（`m67_oos_probe` 是 import 之後才
  `os.environ.update()` 換 arm，import-time gate 會靜默變成假陰性）、壞值 fallback
  不 raise（跑在 grader 的 `solve()` 裡，例外賠掉整案）。
- gate 三項並聯：`ICCAD_M80_TIER` ∧ `_effective_cores_hi() >= 40` ∧ `n > ICCAD_M80_MIN_N`。
  **用 `_effective_cores_hi()`（unknown→0，fail-CLOSED）**，不可用
  `_effective_cores()`（unknown→9999，那是 tier-3/4「低核才開」的安全方向）。
- gate 讀在 `ICCAD_ADAPTIVE_POOL=0` 的 early return **之前**（與 M72/M76 同紀律），
  否則會污染離線錨、M53 L1/L3 與 probe 自己的 `full` 端點。
- M80 索引**要**吃 `_m71_env()`（cloud 就是在 M71 overlay 下量的）——
  這點與 M76 escape tier 正好相反，是 gate A 專門驗的一項。

### 🔑 零 cache 作廢：append 在出貨前綴之後

CLAUDE.md 原本假設「進 `_PROFILES` 就要走完整 regen 鏈」。**那只在塞進出貨 41 隻時成立。**
四顆離線 cache 的簽章全都錨在 `_PROFILES[:_M55_BASE_LEN]`（M73 那次專門改的）：

| cache | 大小 | append 後 |
|---|---:|---|
| `audit_cache{,_ship,_esc}.pkl` | 11 MB×3 | ✅ 有效 |
| `m67_oos_cache{,_c48}.pkl` | — | ✅ 有效 |
| `m77_oos_audit.pkl` | 49 MB | ✅ 有效 |
| `m79_knob_cloud.pkl` | 35 MB | ✅ 有效（簽章根本不含 `_PROFILES`） |

而且**進前綴一點好處都沒有**：48 核上出貨池已經是 35 隻/每帶（tier-5 還原 M42、
tier-3 因 `_effective_cores() > 16` 關閉、tier-4 只在 ≤8 核開），唯一還活著的前綴剪法是
M41 的 **content-based** swap 過濾，而 cloud 生成時就排除了 `ORDER_SWAP`/`ORDER_MOVE`
⇒ **M80 與整條 drop-常數推導鏈零交互**。省下 30-35 分鐘 audit + 兩顆 OOS cache 重建。

## 教訓

1. **「單獨死不代表聯合死」**——ledger 裡所有「某旋鈕封卷」的判定，只對單軸掃描成立。
   要找洞，去找**沒有被聯合抽樣過的參數組合**，不是再掃已知旋鈕的值。
2. **加大提案數的回報還沒飽和**：R 從 128 到 256，K=8 的 held-out 從 0.791% 漲到 1.000%，
   per-case oracle 從 +2.025% 漲到 +2.649%。R=512 沒試過。
3. **ML 那半邊被釘得更死**：cloud 變大讓 oracle 漲到 +2.649%，但 LOO 預測器完全沒動
   （global +0.166%、band +0.051%、knn5 +0.127%）⇒ **oracle 與可預測值的差距是變寬不是變窄**。
   M79 的 RED 在更大的搜尋空間下更穩固。
4. **「加候選預設有害」（M78）不是普遍律**：M78 量到在 anchored 路徑加候選是 −0.18%、
   在泛用 `item_candidates` 是 +0.36%，結論是「出貨的候選集合是調過的」。M80 加的是
   **整隻 profile**（portfolio 層），不是 packer 內部的候選位置——proxy 仲裁在異質候選上
   是 oracle-perfect（M76/M77 驗過），所以這一層加東西是弱單調的。**兩件事不要混為一談。**
5. **但 `hmin` 耦合仍然真實**：proxy 的 `hmin` 是**整池**的 min HPWL，新候選壓低它會等比
   放大所有候選的 hpwl 項卻不動 `area/Â` 項 ⇒ 既有候選之間的排序**可以翻**。實測 s1/s2
   各有 2-5 案變差。所以 `m67_oos_probe` 的 M80 arm **刻意不放進 strict「永不變差」分支**。

## 誠實範圍

- OOS 的 shipped 基準是 **48 核池形狀**（`--force-cores 48` / `--cores 48`）。M76 量過
  形狀差 2.7 倍，16 核形狀的數字不可比。
- **M80 與 tier-5 共用同一個賭注**（評分機有效並行度 ≥40）。若賭輸，兩者一起不觸發，
  增益歸零 —— 但**不會變負**（fail-CLOSED）。
- dRF 用 `m77_oos_audit.pkl` 的逐 profile dt（11 worker 併發下量，與 `profile_audit`
  同條件）+ 新向量實測 dt 推。s1 的 dRF（+0.287%）比 s2（+0.011%）大得多，是抽到的
  案子不同造成的 max-setter 差異；**報告與 K 的選擇一律用較保守的 s1**。
- R=256 仍是有限抽樣、提案分布偏向已知好區域（一半是出貨 profile 的 1-3 knob 擾動）
  ⇒ per-case oracle 的 +2.649% 是**該提案分布下**的上界。
- 5-fold CV（1.000% @K=8）是**同語料**的，只交叉驗證「挑哪些向量」；真正的判定是
  OOS 240×2。兩者符號一致、OOS 反而更大（語料更難、headroom 更多，與 M71 同型態）。

## 驗證

| 項目 | 結果 |
|---|---|
| `build_cloud` prefix-stability（128 vs 256 前綴） | 逐字相同，8 隻與 `m79_greedy.txt` 一致 |
| `m80_tier_probe selftest --sample s1 --cores 48`（K=0） | **PASS** `1.555854672` 與 m77 逐位相同、0/240 winner 差異 |
| OOS build 失敗數 | s1 2856/2856、s2 2880/2880，**0 fail** |
| `inset` 曲線 vs `m79_knob_cloud_probe greedy` | K=8 皆 `1.266623425`（逐位） |
| 官方 eval 預設（16c，tier 惰性） | `1.2934610352`，**0 movers** vs `results_M74_default.json` |
| 官方 eval 48c、`ICCAD_M80_TIER=0` | `1.2934610352`，**0 movers** |
| 官方 eval 48c、tier ON | **`1.2666234251`** = 離線 K=8 **逐位相同**；56 好 / 2 壞、**100/100 feasible** |
| `m80_tier_gate.py` V1-V6 | **ALL PASS** |
| `m67_oos_probe restore --arm m80 --force-cores 48` Gate A | **ALL PASS**（含 cores gate 39c→13 / 40c→21，tier-5 已隔離） |
| 同上 Gate B（in-set 100，真 wrapper） | shipped 1.293461 → **1.266623**，58/100 movers（與官方 eval 逐案一致） |
| 同上 OOS 240 @48c（**真 wrapper**） | shipped `1.555855` → **`1.523604`**，131 好 / 3 壞、**0 infeasible** |
| `regression_suite.py`（八項） | **OVERALL: ALL PASS**（856s）—— m48 / rf / m49×3 / m47b / m67g tier-5 / **m80 tier** |

**同一個數字 `1.266623425` 由四條互相獨立的路徑得到**：離線 greedy（`m79_knob_cloud_probe`）、
離線重算（`m80_tier_probe inset`）、官方 evaluator、`m67_oos_probe` 的真 wrapper Gate B。
**OOS 那邊也對上了**：真 wrapper 跑完 240 案得 `1.523604`，與 `m80_tier_probe score` 的
K=8 值 **`1.523603590`** 相同 ⇒ 離線模擬器在 K>0 也是忠實的，不只在 selftest 的 K=0。

> ⚠️ **兩支工具的「gain %」分母不同，不要混用**：`m67_oos_probe._ab_report` 印
> `(S/R − 1)`＝**+2.117%**，`m77_oos_probe`／`m80_tier_probe` 印 `(1 − R/S)`＝**+2.073%**。
> 同一個變化、不同基準。**本報告一律用後者**，因為 OOS NET bar（0.30%）就是在 m77 那個
> 慣例下定義的，而且在這個量級它是**較保守**的那個。以往的 arm 都在 0.1-0.3% 量級，
> 兩者差 <0.001pp 所以沒人注意；到了 2% 就差 0.044pp。

## 產出物

- `optimizer_constructive.py` — `_M80_EXTRA`/`_M80_IDX`/`_M80_CORES_MIN`/`_m80_active()`
- `m80_vectors.json` — 12 隻的貪婪順序（出貨取前 8），`m80_tier_gate.py` V5 逐字斷言
- `m80_tier_probe.py` — `build|score|selftest|inset`（重用 `m77_oos_audit.pkl`）
- `m80_tier_gate.py` — V1 惰性 / V2 blast radius / V3 前綴不動 / V4 fail-closed /
  V5 向量身分 / V6 可達性+M71 overlay
- `m79_knob_cloud_probe.py` — `KMAX` 可調、新增 `dump` mode
- `m67_oos_probe.py` — `m80`/`m80big` arm（順帶修掉 M76 起就壞掉的 `m55`/`m55x`
  Gate A：它用 `len(_PROFILES)` 與寫死的 `4` 描述 M72 tier，M76 append 41 隻 escape twin
  之後就變成在斷言整條尾巴。改錨 `_M55_IDX`）
- `m67g_tier5_gate.py` — `_ISOLATE` 把 M80 釘 OFF，讓 V1/V3 繼續只驗 tier-5
- `regression_suite.py` — 第八項 gate
- 日誌：`m80_cloud256_run.txt`、`m80_greedy256.txt`、`m80_loo256.txt`、
  `m80_build_s{1,2}.txt`、`m80_score_s{1,2}.txt`、`m80_inset.txt`
- 結果：`results_M80_oos_s{1,2}_c48.json`、`results_M80_{default,c48_off,c48_on}.json`、
  `results_M72_ab_m80_0_inf.json`（m67 arm 的真 wrapper OOS）
- commit `0f40fda`（10 檔、+1530 行）

## 送件影響（**必讀**）

`build_submission/` 那顆是 **M74**，**不含 M80**。評分機是 48 核 ⇒ 送舊包等於白丟 **−2.075%**。
Final（8/21）要送就必須重打包，而且**所有錨要一起換**：`make_submission._ANCHOR`、
`m67c` 的 `ANCHOR`/`ANCHOR48`/`_SOURCES`、`FINAL_LINUX_VERIFY_RUNBOOK.md` 內的預期逐位值
（`final48` 那輪會從 M73 的 `1.295547821428148` 變成 M80 的 48 核值，因為 tier 開著）。
⚠️ GPU 機 WSL 的 `verify_final_tar.sh` 仍然是唯一卡住的一關，這台沒有 WSL/Docker。
