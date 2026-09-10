# M76 追問回覆 — escape tier 的 +0.10% 是純品質；tier-5 的覆蓋範圍到哪

**日期**：2026-08-01 · **回覆對象**：組員 · **來源**：`M76_REPORT.md`、`M67E_REPORT.md`、`optimizer_constructive.py:757-838` 實測

---

## 你們問的

> +0.101~0.107% 是純品質增益，還是已經扣掉 wall 代價之後的淨值？
> 如果是純品質 → tier-5 已經把 escape 想救的案子吃掉九成以上；
> 如果是淨值 → 毛的品質增益可能還不小，只是被 wall 吃掉。
> （目的：想評估往池子裡加新 profile 的空間。）

**一句話**：是**純品質**（分支 A），但吸收率是 **2/3 不是九成**；而且對你們真正的目的來說，
「tier-5 覆蓋到哪」問錯了問題 —— **48 核上池砍已經幾乎全部關掉了**（見 §肯定-5）。

---

## 肯定（已量測，有數字可查）

### 1. +0.101~0.107% 是純品質，wall 代價尚未扣

本地 eval 強制 `RF=1.0`（`iccad2026_evaluate.py:924-940`），所以任何 OOS 分數天生就是
RF-free 的 fiction。`M76_REPORT.md` 的表格三欄本來就是分開的：

| arm | 來源集 / 帶別 | OOS@48c 形狀（**品質**） | dRF@48c（**wall**） | **NET** |
|---|---|---|---|---|
| `m73` | 組員 (2,22,23,25) / 全帶 | +0.105% | +0.088% | +0.017% |
| `m73big` | 組員 (2,22,23,25) / n>100 | +0.101% | +0.020% | +0.081% |
| `m73x` | 我方 (21,23,2,22) / n>100 | +0.107% | +0.020% | **+0.087%** |

bar 是 0.30%，所以品質那一欄自己就已經 RED，wall 只是再削一刀。

### 2. tier-5 吃掉的是 2/3，不是九成

同一個 arm、只換池形狀：

| | `m73x` OOS 增益 |
|---|---|
| 16 核形狀（heavy 池 13，M42 砍掉 22 隻） | **+0.294%** |
| 48 核形狀（heavy 池 35，tier-5 把 22 隻放回） | **+0.107%** |

吸收率 = (0.294 − 0.107) / 0.294 = **63.6%**。

### 3. 比例才是重點：tier-5 自己比整條 escape 軸大 4.5 倍

| OOS 240 shipped 基準 | |
|---|---|
| 16 核形狀 | 1.576749 |
| 48 核形狀 | **1.555855**（**−1.325%**） |

那 −1.325% 就是 tier-5 的 OOS 價值（先前在 80 案 held-out 上量到 +2.289%，加權到
240 案量級吻合）。整條 escape 軸最好的變體是 +0.294%（而且那是沒有 tier-5 時）。

### 4. tier-5 的覆蓋範圍（照 code 定義，不是印象）

`optimizer_constructive.py:795-800, 832-834`：只跳過 M42 的 `_BIG_REDUNDANT_IDX`（22 隻）、
只在 `n > 100`、只在 `_effective_cores_hi() >= 40`（fail-CLOSED，unknown→0）。
**不碰** tier-3 / tier-4 / M41 swap 砍 / `_band_env()` 的 REFINE band-cut。

### 5. 🔑 但這問錯了 —— 48 核上池砍已經幾乎全關

剛在 tree 上實測 `_pool_indices()`（M74 現況）：

| n | 12 核 \|P\| | 48 核 \|P\| |
|---|---|---|
| 30 | 35 | 35 |
| 50 | 35 | 35 |
| 80 | 20 | **35** |
| 105 | 13 | **35** |
| 120 | 13 | **35** |

48 核上**每一帶都是 35 隻、完全相同**。三個 tier 同時失效：

- tier-5 關掉 `n>100` 的 M42（22 隻）
- M74 把 tier-3 從 universal 降級成 `_effective_cores() <= _M45_MID_CORES_MAX (=16)`
- tier-4 只在 `_effective_cores() <= 8`

⇒ **48 核上唯一還活著的池砍，是 M41 的 6 隻 swap/move**（41 − 6 = 35）。

### 6. 「不比現任最慢那隻慢就免費」成立，且有實測支撐

- 48c wall = max-setter，**100/100 案**（`M67E_REPORT.md` §2，`Σ/48` 只有 max 項的 3-27%）
- OLS（40 隻 REFINE-free 案）`meas = a·W12 + b·|P| + c` → **a=0.9997、b=2.45 ms/profile、c≈0**
  ⇒ 每多一隻 profile 的序列成本只有 **2.45 ms**
- M47 後 `_proxy_metrics` ≈ 2.5 ms/隻，ΣPT 逐帶 0.03-0.09s，遠低於 max 項

### 7. 免費的邊界是 `c* = Σdt / max dt`

restore 池逐帶實測：

| band | \|ship\| | \|restore\| | dW@48c | **c\*(restore)** |
|---|---|---|---|---|
| (0,40] | 35 | 37 | +0.00% | 27.2 |
| (40,60] | 35 | 38 | +0.00% | 23.9 |
| (60,100] | 26 | 37 | +0.00% | 24.2 |
| (100,110] | 13 | 33 | +0.00% | 22.0 |
| (110,inf] | 13 | 34 | +0.00% | **23.4** |

加一隻 `dt <= max-setter` 的 profile，`c*` 最多 +1。只要 `c* <= 有效並行度`，wall 就不動。

### 8. 加進來的品質是全額拿得到的

- proxy 自 M13 起 = per-case oracle ceiling ⇒ selection 不是瓶頸
- M76 再補一個獨立證據：41 隻 knob-off 的 **full-union 混合池 −0.388%，贏過 2-way oracle −0.340%**
  ⇒ proxy 在異質候選的聯集上找得到兩個端點都沒有的解

---

## 推測（投影 / 外推，未實測）

### 1. 48c 的一切都是投影

本機只有 12/16 核。dt 是 12 核 11-worker 量的；48 核形狀是 `ICCAD_ADAPTIVE_CORES=48`
模擬**池組成**，不是真的在 48 核機器上跑。wall 的「+0.00%」來自 M67-E 的模型。

### 2. 餘裕沒有 `48 − 23 = 25 隻` 那麼多

偵測核數是有效核的**上界**（本機 16 邏輯核 ≈ 10 有效，比值 0.63）。
若 Beta 的 48 核 → 有效 ~30，重帶 `c* = 23.4` 的餘裕就只剩 **~6 隻**。

⇒ 我方對「加新 profile 空間」的估計：**個位數，且必須挑 dt 低於 max-setter 的便宜貨**。

### 3. escape 殘下的 0.10% 不是均勻殘留

in-set 的可回收量 17 案、加權 0.004397 分數點，其中 **82.9% 集中在 5 案**
（98/83/92/94/82，和你們量到的 98/79/83/92/94 幾乎同一批）。
推測 OOS 也是這個形狀 ⇒ 想吃這 0.10%，要針對那幾案的結構，加通用 profile 沒用。

### 4. 共同賭注（這條結論最大的條件依賴）

若評分機的**有效並行度 < 40**：tier-5 不觸發 → 重帶池回到 13 → escape 回到 +0.29%，
**但同時「加 profile 免費」整個作廢**（那時 wall 是 `Σ/cores` bound，12 核實測 dRF +4.5%）。
兩件事共用同一個賭注，且方向相反。

---

## 否定（別這樣讀 / 別這樣做）

### 1. ❌「+0.10% 是淨值」

別再扣一次 wall。淨值是 **+0.017~0.087%**。

### 2. ❌「毛品質增益可能不小，只是被 wall 吃掉」

在評分機形狀上這是反的。48c 的 dRF 只有 +0.020~0.088%，**重帶恰 +0.000%**：

| band | `dt_esc / dt_on` p50 | | | |
|---|---|---|---|---|
| | #2 | #22 | #23 | #25 |
| (60,100] | 0.987 | 1.061 | 1.023 | 1.036 |
| (100,110] | 0.905 | 1.022 | 0.929 | 0.976 |
| **(110,inf]** | **0.934** | **0.911** | **0.906** | **0.915** |

重帶的 knob-off 副本比它的 knob-on 本尊**更快**，不可能頂掉 max-setter
（順帶回答你們未解的第 3 項：knob-off 逃生口在 48 核**不會**變成 max-setter）。

「被 wall 吃掉」是 **12 核**的形態（加權 dRF +4.504%）—— 那不是評分機。

### 3. ❌「九成」

是 **2/3**。機制是 tier-5 放回的 22 隻 knob-**ON** profile 已經救掉同一批案子，
**同一份分數不能算兩次**。escape 與 tier-5 是**替代品**，不是互補品。

### 4. ❌「48c 加 profile 免費 ⇒ 儘管加」

免費結論限 **`dt <= 當前 max-setter`**。M41 砍掉的 OS16 / OM8 是 11-12s，遠超 max-setter；
`full` 欄的 +53~60% 就是它們 + REFINE 造成的。那 6 隻**不能**算成「池子還有空位」。

### 5. ❌ 用 in-sample 挑新 profile

`m73x` vs `m73big` 是同機制、同 gate、只換來源集的乾淨對照：

| | in-set 100 | OOS@16c | OOS@48c |
|---|---|---|---|
| `m73x` − `m73big` | **+0.127pp** | **+0.006pp** | **+0.006pp** |

**轉移率 ≈ 5%**。這是 M72 / M74 / M75 之後第四次。

### 6. ❌ 用 16 核形狀量 OOS

in-set 100 在兩種形狀下**逐位相同**（1.293461 / m73x 1.289345 / 5 movers），
OOS 卻差 **2.7 倍**（+0.294% → +0.107%）。任何與 adaptive tier 有交互的機制，
OOS 一律要 `m67_oos_probe.py --force-cores 48`。

---

## 對「池子還能不能再加東西」的結論

1. **沒有「還原被砍的」空間了** —— 48 核上池子已是 35/41、每帶相同，
   剩的 6 隻是必須留的巨獸（M41 在 48c 仍是主要 RF 來源）。
2. **要加只能是新造的異質候選。** M76 的 full-union 天花板 −0.388% 證明異質候選確實還有肉，
   只是 41 隻 knob-off 副本買不起（ΔRF +57%）—— 天花板存在，但要挑便宜的路徑上去。
3. **空間大小**：推測個位數隻（重帶 `c*` 23.4，有效並行度可能只有 ~30）。
4. **現成的 gate 就是 `m77_ml_candidate_probe.py`**：吃任何 optimizer 的官方 results json，
   當成第 42 隻候選接進池子走 proxy 仲裁，直接吐 **portfolio delta（bar 0.05%）+ dRF@48c**。
   「加新 profile 值多少」與「ML 候選值多少」是同一個問題、同一支工具。

---

## 復現

```powershell
# 逐帶池組成（本文 §肯定-5）—— 實測輸出 [(30,35),(50,35),(80,35),(105,35),(120,35)]
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" -c "import os,importlib.util; [os.environ.pop(k) for k in list(os.environ) if k.startswith('ICCAD_')]; spec=importlib.util.spec_from_file_location('oc','optimizer_constructive.py'); oc=importlib.util.module_from_spec(spec); spec.loader.exec_module(oc); os.environ['ICCAD_ADAPTIVE_CORES']='48'; print([(n,len(oc._pool_indices(n))) for n in (30,50,80,105,120)])"

# OOS 兩種形狀（本文 §肯定-2）
python -u m67_oos_probe.py restore --arm m73x --pool0-lo 0                  # 16 核形狀
python -u m67_oos_probe.py run     --force-cores 48                          # 48 形狀基準
python -u m67_oos_probe.py restore --arm m73x --pool0-lo 0 --force-cores 48  # 48 形狀

# 48c wall / dRF（本文 §否定-2）
python -u m76_escape_probe.py wall --min-n 100
```

參考：`M76_REPORT.md`（完整 M76）、`M67E_REPORT.md` §2/§5（48c 結構與 c\*）、
`M77_ML_GATE_NOTE.md`（portfolio delta gate）。
