# M77 — 給 ML-as-placer 路線的兩點回報（2026-08-01）

> 針對組員 `b7460d3`「目前計畫.md」。兩件事都是我方資料才看得出來的，
> 而且都會改變你們的判準。工具已經建好、驗過，隨時可用。

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
- 目前是 **in-set 100 案精確**（用 `audit_cache_ship.pkl` 的 41 隻 × 100 案）。
  **OOS 240 要另外跑**——給我 240 案的 results json 我再接。
- ⚠️ **OOS 一定要用評分機的池形狀跑**。M76 實測：同一個機制在 16 核與 48 核形狀下
  in-set **逐位相同**，OOS 卻差 **2.7 倍**（+0.294% → +0.107%），因為 48 核上
  tier-5 會把 22 隻 profile 放回 n>100，把增益吃掉。只看 in-set 會完全看不到。

## 4. 另外三點

- **`import torch` 的 DQ 風險你們列為瓶頸 4，這個順序是對的。** 我方現行包是純
  Python + `bin/` 二進位，沒有任何第三方相依。Day-1 就驗 numpy-only 前向是正確的。
- **M76 剛把「escape tier」那條軸關掉**（48 核形狀 OOS 只剩 +0.10%，被 tier-5 吃掉）。
  你們的 M73 +0.288% 是 12 核數字，且你們 repo 沒有 tier-5，看不到抵銷。細節見
  `M76_REPORT.md`。⇒ **古典線這邊沒有剩餘增益了**，全力 ML 的資源判斷我同意。
- **M74 送件包已備妥並驗過**（`regression_suite` 7/7、官方指令逐位
  `1.293461035226291`、Linux bundle 已建），只差 GPU 機的 WSL 那一關。
  不管 ML 成不成，8/21 都有東西可送。
