ICCAD 2026 FloorSet Baseline README
====================================

目的
----

這份 README 用來說明目前 `iccad2026contest` 資料夾內 baseline 程式的用途、版本命名方式、測試指令，以及目前最佳版本的管理方式。

目前主線
--------

目前正式主線為：

```text
my_optimizer.py
my_optimizer_best.py
```

這兩份檔案應該保持一致。

其中：

```text
my_optimizer.py       官方 evaluator 預設測試的正式版本
my_optimizer_best.py  目前最佳版本的備份
```

目前最佳版本是由 `my_optimizer_legal_baseline_v4.py` 發展而來。

目前最佳版本狀態
----------------

目前最佳版本特性：

```text
Feasible: 100 / 100
Hard constraints: passed
Runtime: about 0.01 s average
```

目前版本已經能通過 100 筆 validation，重點是保持合法性，不要破壞：

```text
no overlap
soft block area within tolerance
fixed-shape dimensions unchanged
preplaced position and dimensions unchanged
```

檔案說明
--------

```text
my_optimizer.py
```

目前正式最佳版。官方 evaluator 測試時主要使用這份。

```text
my_optimizer_best.py
```

目前最佳版備份。當新的版本確認比目前版本更好時，才同步更新這份。

```text
my_optimizer_results.json
```

目前正式版最近一次 evaluate 的結果。

```text
my_optimizer_best_results.json
```

目前最佳版結果備份。

```text
my_optimizer_legal_baseline_v1.py
```

第一版 legal baseline。策略是把所有可移動 block 排成一條水平長列。分數很差，但可作為最基本合法版本參考。

```text
my_optimizer_legal_baseline_v2.py
```

第二版 shelf packing。策略是將 blocks 分成多排放置，降低 bounding box area。

```text
my_optimizer_legal_baseline_v3.py
```

第三版 grouping-aware packing。策略是把同 cluster/group 的 movable blocks 組成 composite item，使組內 blocks 互相貼齊，以降低 grouping violation。

```text
my_optimizer_legal_baseline_v4.py
```

第四版 boundary-aware ordering。策略是在 item 排序時提高 boundary constrained blocks 的優先度，使其更容易被排到外側位置。

```text
my_optimizer_legal_baseline_v5.py
```

第五版 boundary-aware candidate scoring。策略是在挑選 shelf width candidate 時，同時考慮 area score 和 boundary violations。實測後 evaluate 表現較不穩，目前未作為正式主線。

版本演進摘要
------------

```text
v1: one-row legal baseline
v2: shelf packing
v3: grouping-aware composite packing
v4: boundary-aware ordering
v5: boundary-aware candidate scoring
```

目前建議正式使用：

```text
my_optimizer.py
my_optimizer_best.py
```

測試指令
--------

格式檢查：

```bat
python iccad2026_evaluate.py --validate my_optimizer.py
```

跑單一 testcase：

```bat
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 99
```

跑全部 validation：

```bat
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

指定輸出檔：

```bat
python iccad2026_evaluate.py --evaluate my_optimizer.py --output my_optimizer_results.json
```

保存 solution：

```bat
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions --output my_optimizer_results.json
```

重新計算固定 solution 的分數：

```bat
python iccad2026_evaluate.py --score my_optimizer_solutions.json --output my_optimizer_static_score.json
```

注意：正式流程較接近 `--evaluate`，因為它會重新呼叫 `solve()` 並計算 runtime。`--score` 主要用來比較固定 layout 品質。

目前版本管理方式
----------------

正式主線固定使用：

```text
my_optimizer.py
```

目前最佳備份固定使用：

```text
my_optimizer_best.py
```

若要開新版本，例如 v6：

```bat
copy /Y my_optimizer.py my_optimizer_legal_baseline_v6.py
```

測試 v6：

```bat
python iccad2026_evaluate.py --validate my_optimizer_legal_baseline_v6.py
python iccad2026_evaluate.py --evaluate my_optimizer_legal_baseline_v6.py --output v6_run1.json
```

若 v6 多次 evaluate 後確認比目前 `my_optimizer.py` 更好，再更新正式版：

```bat
copy /Y my_optimizer_legal_baseline_v6.py my_optimizer.py
copy /Y my_optimizer_legal_baseline_v6.py my_optimizer_best.py
copy /Y my_optimizer_legal_baseline_v6_results.json my_optimizer_best_results.json
```

判斷版本是否變好的標準
----------------------

第一優先：

```text
Feasible 必須維持 100 / 100
```

第二優先：

```text
多次 --evaluate 的 Total Score 較低且穩定
```

第三參考：

```text
Avg Cost
Runtime
saved-solution static score
```

目前觀察到 `Total Score` 會因 runtime factor 有波動，因此不要只看單次 evaluate 結果。建議同一版本至少跑數次，再比較平均、最好、最差情況。

目前不建議做的事
----------------

```text
不要直接使用 random placement
不要在正式版加入大量 simulated annealing
不要讓 runtime 明顯變慢
不要為了降低 static score 犧牲 evaluate 穩定性
不要破壞 100 / 100 feasible
```

目前建議下一步
--------------

從目前最佳版開新版本：

```bat
copy /Y my_optimizer.py my_optimizer_legal_baseline_v6.py
```

v6 建議方向：

```text
保持 v4 的速度
不要沿用 v5 太重的 boundary candidate scoring
嘗試輕量化的 boundary/grouping ordering
觀察 large testcase 的 Total Score 是否下降
```

目前正式提交前確認流程
----------------------

```bat
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 99
python iccad2026_evaluate.py --evaluate my_optimizer.py --output final_check_results.json
```

若結果符合：

```text
Feasible: 100
程式不 crash
runtime 不明顯變慢
Total Score 在目前最佳範圍內
```

即可把 `my_optimizer.py` 視為目前正式提交版本。
