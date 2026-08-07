# M77 — 給 ML-as-placer 路線的回報（2026-08-01，08-02 補第 4 節）

> 針對組員 `b7460d3`「目前計畫.md」。都是我方資料才看得出來的，
> 而且都會改變你們的判準。工具已經建好、驗過，隨時可用。
> **2026-08-02 更新：OOS 240 的介面也接好了，見第 4 節——那才是 ship 判定的那把尺。**

## 1. kill gate 的度量錯了：solo total 不能決定 ML 的價值

計畫寫死的 gate 是 **「ML-only weighted total 隨資料量單調下降，rung 2 必須 < 1.6，
否則 ML-as-placer 判死」**。但同一份文件的鐵律是部署形態 =
**每案 `max(ML, 古典)`，由 baseline-free proxy 仲裁**。

這兩個是不同的問題，而且**不是單調相關**。我用手上兩個真實候選量了：

| 當成外部候選餵進 41 隻池 | solo total | **portfolio 價值** |
|---|---:|---:|
| M74 自己的 portfolio 輸出 | **1.293461**（最好的 solo） | **恰好 +0.000%** |
| M71 旗標全關的 portfolio | **1.337769**（差 3.4%） | **+0.340%** |

**solo 分數比較好的那個價值 0，比較差的那個價值 +0.34%。**
原因很直觀：portfolio 只在乎「你在**哪些案子**上贏」，不在乎你的平均。
一個 solo 2.0 但在 10 個重案上贏的模型是有價值的；一個 solo 1.6 但一案都沒贏的
模型價值恰好 0。

⇒ **rung gate 應該換成「ML 候選在 proxy 仲裁下對 M74 的加權增益」**，
bar 沿用我方 `profile_vs_portfolio.py` 的 **0.05%**。
這個 gate 比 `< 1.6` **寬鬆得多**（爛模型也可能過），也**嚴格得多**
（不贏任何案子的好模型會被擋下）——兩個方向都比原本的 gate 正確。

### 為什麼可以相信 proxy 撈得到

兩個獨立證據，都是 2026-08-01 實測：

1. **M76 full-union**：41 隻 knob-ON profile + 41 隻 knob-OFF 雙胞胎
   （兩組 hpwl/area 尺度明顯不同的異質候選）丟進同一個池，proxy 選出來的結果
   **恰好等於逐案 2-way oracle**（1.288443844，兩者逐位相同）。
2. **上表第二列**：17 個該贏的案子，proxy **17/17 全撈到、0 誤選**，
   selection efficiency **100.0%**。

⇒ **selection 不是瓶頸**，即使候選來自完全不同的生成機制。
你們不需要 ML 「總是」比較好，只需要它「有時候」比較好。

## 2. fallback 錨過期了 2.5%

計畫寫「**最壞情況 = 1.3265，不會更差**」——那是 pre-M71 的 M67-G。

| 版本 | local100 | 狀態 |
|---|---:|---|
| M67-G | 1.3265 | 你們計畫裡的 fallback 錨 |
| M71（已上傳的 Beta 包） | 1.3054 | |
| **M74（tree 上的現況）** | **1.293461** | 已過 `regression_suite` 7/7 + 官方指令逐位驗證 |

用 1.3265 當下檔保護，等於自願放棄 **2.5%**。ladder 的原點、
`max(ML, 古典)` 的古典側、以及所有 headroom 百分比都該改用 **1.2935**。

（順帶：計畫表裡「隊友 M74 local 1.2935」已經列對了，只是鐵律那段沒跟上。）

## 3. 工具：`m77_ml_candidate_probe.py`

**輸入 = 一份官方 results json**（`iccad2026_evaluate.py --output` 的產物，
你們的 `optimizer_ml_transformer.py` 跑一次就有）。

```powershell
python m77_ml_candidate_probe.py score <ml_results.json> --cores 48 --dt <推論秒數>
python m77_ml_candidate_probe.py selftest      # 不變式自檢
```

輸出三個數字：

| 數字 | 意義 |
|---|---|
| **portfolio delta** | ML 實際值多少 —— **這是 gate** |
| **oracle delta** | 完美 selection 下值多少 |
| **efficiency** | 兩者比值。低 ⇒ proxy 的問題；高但 delta 小 ⇒ 模型的問題 |
| **dRF@48c** | runtime 代價。48 核上 wall = max-setter（M67-E 實測 100/100） ⇒ **dt 低於現任 max-setter 的候選是免費的** |

外加：ML 在哪幾案贏、proxy 有沒有撈到、有沒有誤選、有幾案 infeasible。

**自檢**：把 portfolio 自己的輸出餵回去必須恰好值 0（0 wins / 0 switches /
+0.000000%）——已 PASS，代表離線選擇器與 wrapper 逐位一致。

### 使用注意

- **`--dt` 要給模型自己的每案推論時間**。results json 的 `runtime_seconds` 是
  **整個 solve 的 wall**，不是單一候選的，直接拿來算 dRF 會得到假數字
  （工具會警告）。推論可忽略就 `--dt 0`。
- 這支是 **in-set 100 案**（用 `audit_cache_ship.pkl` 的 41 隻 × 100 案）。
  **OOS 240 已於 2026-08-02 接好**，見下面第 4 節（另一支工具）。
- ⚠️ **OOS 一定要用評分機的池形狀跑**。M76 實測：同一個機制在 16 核與 48 核形狀下
  in-set **逐位相同**，OOS 卻差 **2.7 倍**（+0.294% → +0.107%），因為 48 核上
  tier-5 會把 22 隻 profile 放回 n>100，把增益吃掉。只看 in-set 會完全看不到。

## 4. OOS 240 介面已接好（2026-08-02）：`m77_oos_probe.py`

第 3 節那支只看 in-set 100。**但 in-set 過關不代表能送**——M76 量到
**in-sample 優勢的轉移率只有 ≈5%**（同機制同 gate、只換來源集：in-set 差 +0.127pp，
OOS 只差 +0.006pp）。所以 ML 候選的 ship 判定必須在 OOS、且在 **48 核池形狀** 下做。

### 兩份樣本，請兩份都跑

| 樣本 | 抽樣範圍 | 用途 |
|---|---|---|
| **s1** | `floorset_lite` **worker_0..9**（seed 67，每 n 2 案、n>100 每 n 4 案） | 與我方所有歷史 OOS 數字**同一批案子**（M67-D/F、M72、M75、M76），可直接跟古典 arm 比 |
| **s2** | **worker_10..19**，同樣抽法，與 s1 **交集 0** | 🚨 **ML 的判定看這份** |

**為什麼要 s2**：s1 抽自 `floorset_lite`，那正是你們的訓練語料。對我方古典調參它是乾淨
OOS，但**對一個在 worker_0..9 上訓練過的模型，s1 是樣本內**。s2 才是誠實的 OOS。
（s1 仍要跑：它是唯一能跟我方古典數字對齊的尺。兩份差很多 = 記憶效應的直接證據。）

### 你們要做的事

1. 讀 `m77_oos_manifest_s1.json` / `m77_oos_manifest_s2.json`（各 240 案，我方產出，
   內含 `keys_md5` 可比對）。每一筆長這樣：
   ```json
   {"oos_id": 0, "key": "worker_1/layouts_1008/L64",
    "file": "floorset_lite/worker_1/layouts_1008.th", "layout": 64, "n": 21}
   ```
   `torch.load(file)` 得到 7-tuple，`d[0][layout][:,0]` 是 area_target（-1 是 padding，
   n = 非 -1 的數量）、`d[0][layout][:n,1:]` 是 5 個 constraint 欄、`d[1]/d[2]/d[3]`
   是 b2b/p2b/pins。
2. 用你們的 placer 跑那 240 案，回一份 json：
   ```json
   {"submission_name": "<model tag>", "sample": "s2",
    "test_results": [{"oos_id": 0, "key": "worker_1/layouts_1008/L64",
                      "positions": [[x, y, w, h], ...],
                      "runtime_seconds": 0.31}]}
   ```
   `key` 或 `oos_id`（`test_id` 也收）都行，**兩個都會拿去跟 manifest 對**，
   對不上直接報錯——避免樣本錯位卻印出一個自信的錯數字。
   **允許部分覆蓋**（沒給的案子就用古典 winner），`runtime_seconds` 請給**模型自己的
   推論時間**，不是整個 solve 的 wall。
3. 丟給我，或你們自己跑：
   ```powershell
   python m77_oos_probe.py score ml_oos_s2.json --sample s2 --cores 48 --dt 0.31
   ```

### 輸出與判準

同樣是 portfolio delta / oracle delta / efficiency / dRF@48c，外加 **NET = delta − dRF**，
聚合用的是 M67-D 那把尺（先對每個 n 取平均、再套官方加權），所以和我方每一個 OOS
數字直接可比。**bar = NET ≥ 0.30%**（M75/M76 對 OOS arm 的既定 bar；in-set 的 0.05%
只是參考，M76 就是死在 +0.10%）。

### 這支工具本身驗過了

- **離線模擬器 = 真 wrapper，逐位**：s1 的 240/240 案 winner positions 與實際跑出來的
  完全相同，兩種池形狀都是——12/16 核形狀總分 `1.576748536`、48 核形狀
  `1.555854672`，與 M76 記錄的錨（1.576749 / 1.555855）一致。
- **零值自檢**：把 portfolio 自己的輸出當候選餵回去 ⇒ 0 wins / 0 switches /
  **+0.000%**（s1、s2 都測）。
- 錯位防護：`key`/`oos_id` 對不上 manifest 就直接退出，不會靜默算出一個自信的錯數字。
- 為什麼要做到逐位：selector 的 `hmin` 是**整池**的 min HPWL，一個 HPWL 很低的 ML 候選
  會把池裡每個候選的 hpwl 項一起重新縮放 ⇒ **池內排序會變**，不是兩兩比較，
  近似法會給錯的答案。

### 順帶量到的兩件事（你們可能用得上）

我方 M74 shipped portfolio 在兩份樣本上的分數：

| | 12/16 核形狀 | 48 核形狀（評分機） | tier-5 的價值 |
|---|---:|---:|---:|
| s1（worker_0..9） | 1.576749 | 1.555855 | **−1.325%** |
| s2（worker_10..19） | 1.586912 | 1.557814 | **−1.833%** |

1. **s2 只比 s1 難 +0.126%（48 核形狀）** ⇒ 兩份語料難度相當，s1↔s2 的差距可以直接
   當成「模型有沒有記住訓練案子」的量測，不會被語料難度混淆。
2. **tier-5 在一批全新的案子上獨立複現，而且更值錢**（−1.83% vs −1.33%）。這是
   M67-F 那個賭注（評分機有效並行度 ≥40）第一次在 s1 以外的語料上被驗證。

## 5. 另外三點

- **`import torch` 的 DQ 風險你們列為瓶頸 4，這個順序是對的。** 我方現行包是純
  Python + `bin/` 二進位，沒有任何第三方相依。Day-1 就驗 numpy-only 前向是正確的。
- **M76 剛把「escape tier」那條軸關掉**（48 核形狀 OOS 只剩 +0.10%，被 tier-5 吃掉）。
  你們的 M73 +0.288% 是 12 核數字，且你們 repo 沒有 tier-5，看不到抵銷。細節見
  `M76_REPORT.md`。⇒ **古典線這邊沒有剩餘增益了**，全力 ML 的資源判斷我同意。
- **M74 送件包已備妥並驗過**（`regression_suite` 7/7、官方指令逐位
  `1.293461035226291`、Linux bundle 已建），只差 GPU 機的 WSL 那一關。
  不管 ML 成不成，8/21 都有東西可送。
