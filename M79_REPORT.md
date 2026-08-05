# M79 — 自建 ML 候選 Gate 0（2026-08-05）

> 問題：M77 替 ML 候選訂了一把尺，但沒有候選會進來。**我們能不能自己做一顆過得了尺的模型？**
> 回答：**不能，而且理由是量出來的。但同一批探測撞到一條古典增益，held-out NET +0.655%。**

## 判定總表

| 分支 | 問的問題 | in-set | held-out | 判定 |
|---|---|---:|---:|---|
| **G0-D 校準** | fp_sol verbatim 值多少 | **+14.410%** | — | 尺正確 |
| **G0-A 完美形狀** | ML 預測 per-block 形狀 | **+0.099%**（12 recipe 版 +0.112%） | — | **RED**（bar 1.0%，差 9×） |
| **G0-B 旋鈕 oracle** | ML 逐案預測一隻 profile | **+2.025%** | LOO 預測器 **+0.166%** | **RED for ML**（上界真、但不可預測） |
| **G0-B′ greedy 固定 profile** | 同一批向量，**不准逐案選** | **+1.576%**（K=8） | **+0.791%**（5-fold） | **GREEN，而且是古典的** |

**pre-registered bar**（開工前登記，未改）= in-set portfolio delta **≥1.0%**，
理由是真正要過的是 OOS NET 0.30%，而上界到實測要付三次折扣（實現率、OOS 轉移、dRF）。

**結論一句話**：ML 在這條路上沒有工作可做——它能碰的兩個軸，一個上界只有 0.099%，
另一個上界雖然有 2.03% 但**逐案的贏家身分不可從 label-free 特徵預測**（LOO 預測器
+0.166%，比「隨便加一隻固定 profile」的 held-out +0.234% 還差）。
而**去掉「逐案」這個要求之後，同一批隨機向量用固定 profile 的形式就拿到 held-out +0.791%**。

---

## G0-D 校準：尺是對的

fp_sol verbatim 當第 42 候選：solo `1.107940199`（與記錄的天花板 1.1079 相符）、
portfolio delta **+14.410%**、selection efficiency **100.0%**、97/100 案贏、
proxy 誤選 0、dRF@48c +0.000%。

⇒ m77 的算術、加權、pool 形狀都正確，後面三個數字可信。

## G0-A 完美形狀 = RED

### 為什麼要量這個

三個 perfect-information 探測已經把 ML 能碰的軸夾住了，而**形狀是唯一沒被量過的**：

| 完美輸入 | 對 portfolio 的價值 | 出處 |
|---|---:|---|
| 完美排序 | +0.005% | M26 oracle-perm |
| 完美位置種子 | +0.001% | M68 ML-seed |
| **完美形狀** | **+0.099%** | **M79（本輪）** |
| fp_sol verbatim | +14.410% | G0-D |

先驗理由很強：形狀是古典線收穫最大的軸（M29-M37 六個 free-aspect 子軸
1.3862→1.3269，−4.3%），而且**我們現在幾乎沒在用形狀資訊**——in-set 100 的
6110 個 movable soft block，label 的 aspect (w/h) 是 p05 0.400 / p50 1.067 /
p95 2.500、**sd(log)=0.530**，只有 **34%** 落在近正方 [0.8,1.25]，而我們對
interior 軟塊的預設是 `SOFT_ASPECT=1.0` 的正方形。**依 boundary code 分層後
sd(log) 幾乎不降**（各層 0.44~0.55）⇒ 我們唯一在用的形狀特徵（LR 2.50 / TB 0.40）
解釋掉的變異接近 0。

### 機制

`constructive_m79.cpp` 新增 `ICCAD_DIMS_FILE`：讀 `id w h` 覆寫 `dims[]`，並把該 block
鎖住，讓 `apply_safe_mib_dims` / `FREE_ASPECT` / `FREE_CLUSTER` / `FREE_ANCHORED` /
`CLUSTER_ASPECT` / `MIB_ASPECT` 全部跳過它。位置、排序、frame、後處理**全部照我們自己的
placer 跑**——與 M26／M68 同構。未設 env ⇒ 逐位等於 `constructive.exe`（600/600 PASS）。

覆蓋率：oracle 模式指定 **6110/7050 = 86.7%** 的 block（加權面積 85.9%），
其餘是 fixed/preplaced（本來就不是我們的自由度）。

**label 的 `w*h` 與 `area_target` 逐位相等（6110/6110，max rel dev 0.0e+00），
100 個 movable MIB group 的 label 形狀也完全相同** ⇒ 直接沿用 label 矩形即可，
不需要（也不應該）再過一次 `sqrt()`，那只會注入 1 ulp 而可能製造出 V_mib。

### 數字

| arm | 覆蓋 | recipe 數 | portfolio delta | 贏的案子 | dRF@48c |
|---|---:|---:|---:|---:|---:|
| **control（不指定形狀）** | 0% | 6 | **+0.000%** | **0/100** | +0.000% |
| scout（`is_fixed` 路徑、零編譯） | 57.5% | 6 | +0.077% | 10/100 | +0.000% |
| **oracle** | 86.7% | 6 | **+0.099%** | 14/100 | +0.000% |
| oracle（recipe 加倍） | 86.7% | 12 | +0.112% | 15/100 | +0.000% |

**control 恰好是 +0.000%（0 wins / 0 switches）**——那 6 隻 recipe 本來就在 41 隻池裡，
重新選一次一分不多 ⇒ **+0.099% 百分之百歸因於形狀資訊**，沒有「多幾隻 pool」的混淆。
recipe 從 6 加到 12 只讓數字動 **+0.013pp** ⇒ 要靠加 recipe 補到 1.0% 是沒有希望的。

### 為什麼這麼小

`m79_bar_spec.py` 的第 2 張表：到 label floor 的總 headroom 是 **14.343%**，
重案的 relative gap 是 10~25%。所以缺的不是「形狀」這種**逐 block 的參數**，
而是**版圖拓撲本身**。這是 M27（greedy 已在 (area,HPWL) frontier）在另一個面向上的重述：
**餵給 packer 的決策已經吃乾了，瓶頸在 packer 的可達集合。**

⇒ **任何「ML 預測形狀 / 排序 / 位置種子」的路線，上界都在 0.1% 量級。封卷。**

## G0-B 旋鈕空間：上界 +2.03%，但那是 oracle

### 設定

在 41 隻 profile 所在的 ~15 維空間隨機抽 **R=128** 個向量（一半是對某隻出貨 profile
做 1-3 個 knob 的擾動，一半是從 per-knob 先驗重抽；**排除 ORDER_SWAP/MOVE**，
它們 5-12s/案，一進池就自己當 48 核的 max-setter，M41 本來也每案都砍）。
12800 次求解，734s。

### 數字

- 逐案 oracle（用真 cost 挑）solo **1.270162**（**比我們的 portfolio 1.293461 還好**）
- portfolio delta **+2.025%**（oracle +2.031%、efficiency **99.7%**、82/100 案贏、
  **proxy 誤選 0**）、dRF@48c **+0.002%** ⇒ **NET +2.023%**
- 128 個向量裡有 **56 個**在某案是贏家 ⇒ 沒有單一向量支配

這個數字本身就值得存檔：它是 M30/M31 那條線（單隻固定新 profile 掃到飽和 ≤0.063%）
的 **32×**，也是 M53 L2 stochastic union oracle（+0.2377%）的 **8.5×**。
機制上它繞開了 M52（decoder 是我們的 C++，容錯帶是整個空間）與 M56
（**加**候選而非**砍**候選，品質零下檔）。

### 但 ML 拿不到：逐案贏家不可預測

Gate-1 預覽，leave-one-out，判準是 **portfolio delta**（不是 solo cost——
M77 的頭條就是兩者不單調相關；本探測第一版正是用 mean solo cost 挑，
結果 global 預測器報 +0.000%，那是判準錯不是發現）：

| LOO 預測器 | portfolio delta | 贏的案子 |
|---|---:|---:|
| global（訓練集上 portfolio 增益最大的單一向量） | +0.166% | 13/100 |
| band（同 n 帶內） | +0.155% | 12/100 |
| knn5（7 維 label-free 特徵最近 5 案） | **+0.091%** | 8/100 |

**三個都輸給「不看案子、直接加一隻固定 profile」的 held-out +0.234%**，
而且 kNN（最「個人化」的那個）最差。這是 **M56 的完整重演**：
winner 身分 case-idiosyncratic，per-case 預測沒有訊號。

⇒ **ML-as-hyper-heuristic 判 RED**，理由不是「沒有上界」而是「上界不可預測」。

## 🏆 G0-B′ 意外收穫：古典線沒有收斂

把「逐案選」這個要求拿掉，同一批 128 個向量用**固定 profile** 的形式貪婪地加進池：

| K | in-sample | **5-fold CV held-out** | dRF@48c | **NET@48c（held-out）** |
|---:|---:|---:|---:|---:|
| 1 | +0.439% | +0.234% | +0.039% | **+0.195%** |
| 2 | +0.758% | +0.271% | +0.039% | +0.232% |
| 4 | +1.106% | +0.459% | +0.050% | **+0.409%** |
| 8 | **+1.576%** | **+0.791%** | +0.136% | **+0.655%** |

- **單一最佳新向量 in-sample +0.439%** —— M30/M31 用同一把尺（`profile_vs_portfolio`，
  bar 0.05%）掃到飽和時，最好的是 **≤0.063%**。這隻是它的 **7×**。
- **held-out 轉移率 50%**（0.791 / 1.576），遠高於 M76 的 source-set 轉移率 ≈5%，
  與 M78 量到的「機制轉移率 76%」同一個量級 —— 因為這是**新候選**不是**樣本內挑來源集**。
- 5 個 fold 挑到的向量高度重疊（#100/#0/#102/#80/#56/#49/#86/#119 反覆出現）
  ⇒ 不是雜訊過擬合。
- **K=4 與 K=8 的 held-out NET 都過 OOS ship bar（0.30%）。**

### 為什麼 M30/M31 會漏掉

它們是**逐 knob 從人工堆疊的 recipe 往外掃**，一個軸一個軸試、低於 0.05% 就停。
隨機**聯合**抽樣會走到座標式貪婪永遠不會造訪的組合。看挑到的向量就知道：
`#100` 同時帶 `BP_WEIGHT=274048`、`CLUSTER_ASPECT=3.39`、`MIB_ASPECT=0.2338`、
`WIRE_MULT=3.273`、`FRAME_SCALES=1.00,1.10,1.25,1.45` —— 其中 `BP_WEIGHT` 向上、
`MIB_ASPECT` 往 tall 側、寬 frame scale 這三條，**在 ledger 裡各自都被判過死**
（「BP_WEIGHT 雙向封卷」「MIB tall 側 +0.027% 低於 bar」「FRAME_ASPECTS 封卷」）。
**單獨死不代表聯合死。**

⚠️ **12 核上這條是負的**（K=8 dRF **+10.614%**、100/100 案被抬 wall）。
它只在評分機的 48 核形狀成立，和 tier-5 共用同一個賭注。要 ship 就得比照 tier-5
做成 cores-gated tier。

## G0-C 規格表（`m79_bar_spec.py`）

1. **配額**：只贏最重的 top-20 案就要湊到 0.30%，每案需要 **0.37%** 的相對改善；
   要湊 1.00% 則需要 1.24%。top-40 幾乎等於全體（96.4% 權重）。
2. **headroom**：到 label floor 共 **14.343%**，top-10 重案握有 8.466%、top-20 握有
   11.723%。重案的 relative gap 10~25%（case 86 是 25.4%）。
3. **dt 預算**：48 核下 wall = max-setter ⇒ 候選的 dt 只要 ≤ 該案現任 max 就**免費**。
   各帶的 min tmax = **0.16s / 0.79s / 1.37s**（n≤60 / 60<n≤100 / n>100）。
   均勻 dt=1.0s 只要 +0.119%，**dt=2.0s 就要 +3.253%**，dt=4.0s +24.952%。
   ⇒ 任何「跑很多次 placer」的方案在 48 核上都不可能划算；也確認了若真要做 ML，
   **只有 numpy 級的毫秒推論付得起**。

## 誠實範圍

- G0-A/G0-B 全部是 **in-set 100**。G0-B′ 的 held-out 是**同一個語料的 5-fold**，
  不是 OOS 240；真 OOS 要走 `m67_oos_probe.py --force-cores 48`（M76 教訓：
  形狀差 2.7 倍）。
- G0-A 的 oracle 用 label 形狀（離線 oracle，與 M26/M68 同類，永不出貨）。
  **訓練訊號未曾使用 fp_sol**——本輪根本沒有訓練。
- G0-B 的 R=128 是有限抽樣；提案分布偏向已知的好區域（一半是出貨 profile 的擾動），
  所以 +2.025% 是**該提案分布下**的上界，不是整個空間的上限。
- G0-B′ 的 dRF 用 `audit_cache_ship.pkl` 的 dt 與新向量實測 dt 推；新向量的 dt 是在
  11 worker 併發下量的，與 `profile_audit` 同條件。
- 5-fold CV 只交叉驗證了「挑哪些向量」，**沒有**交叉驗證「cloud 本身是在這 100 案上生成的」
  ——不過 cloud 的生成完全不看案子（純隨機 + 出貨 profile 擾動），這一層沒有洩漏。

## 驗證

| 項目 | 結果 |
|---|---|
| `constructive_m79.exe` off-path（未設 `ICCAD_DIMS_FILE`）vs `constructive.exe` | **600/600 (case,recipe) 逐位相同 PASS** |
| `m77_ml_candidate_probe.py selftest` | **PASS**（0 wins / 0 switches / +0.000000%） |
| G0-D 校準落點 | `1.107940199`，與記錄的 1.1079 相符 |
| 所有產出的 results json | 100/100 `is_feasible` |
| 出貨檔 | **`constructive.cpp` / `constructive.exe` / `_PROFILES` 一個字都沒動** ⇒ 三顆 audit cache 全部有效，無需重跑 regen 鏈 |

## 產出物

- `constructive_m79.cpp` / `.exe` — `ICCAD_DIMS_FILE`（gated，off = 逐位相同）
- `m79_shape_oracle_probe.py` — `coverage|calib|scout|oracle|control|offpath`
- `m79_knob_cloud_probe.py` — `run|greedy|loo`（快取 `m79_knob_cloud.pkl`，
  key = `(case, profile-hash)` ⇒ 加大 R 只付新向量的錢；簽章釘 exe md5 + overlay 常數）
- `m79_bar_spec.py` — 規格表
- 資料：`m79_cache.pkl`、`m79_knob_cloud.pkl`、`m79_fpsol_verbatim.json`、
  `m79_shape_{scout,oracle,plain}_{oraclepick,proxypick}.json`（6 recipe；`plain` 是對照組）、
  `m79_shape_oracle_r12_{oraclepick,proxypick}.json`（12 recipe 穩健性版）、
  `m79_knob_cloud_oraclepick.json`、`m79_dims/`
  ⚠️ `m79_shape_oracle_proxypick.json` 這個檔名在 6-recipe 版沒有保留（被 12-recipe 那輪
  覆蓋後改名），要重生就跑 `m79_shape_oracle_probe.py oracle`（快取還在，秒回）
- 日誌：`m79_scout.txt`、`m79_oracle.txt`、`m79_control.txt`、`m79_barspec.txt`、
  `m79_knob_cloud.txt`、`m79_greedy.txt`、`m79_loo.txt`
