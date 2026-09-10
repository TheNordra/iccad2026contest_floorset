# M66：M56 per-case CV 等價類補洞 — RED（不對稱屬實、分數後果零）

**日期**：2026-07-20　**工具**：`m66_equiv_cv.py` + `results_M66_equiv.json`（cost 擴充寄生 `m56_cache.pkl`，360→2713 entries）
**定位**：cache-only 純分析（RF 軸，非 quality）；無新 C++ run、未動任何 shipped 檔案；`m56_percase_oracle.py` 一行未改。

## 1. 前提（code 已驗證的方法論不對稱）

M56 Phase B 的 CV 三處判準不一致（`m56_percase_oracle.py`）：

- `loo_J`（:459-469）：train-LOO J\* 要求 `select(...) != winner` 即 fail —— **index 完全相同**才算 preserved；
- `keep_set`（:444-450）：kNN 聯集只收 singleton `{winner(t, cores)}` label；
- OOS gate（:493-496）：接受任何 **cost 相等（rel 1e-9）** 的替代 profile。

即 train 側過嚴（index-identity）、test 側較寬（cost-equality）——若 winner 有等 cost 替身，原 CV 可能因此低估模型可行性。

## 2. 方法

以等價類 **E(case, regime) = {k ∈ chain_pool(case, cores) : |cost(ci,k) − cost(ci,winner)| ≤ 1e-9·max(1,|cost_w|)}**（tol 逐字同 OOS gate）取代單一 label，套在 **LOO 接受判準**（`sel ∈ E`）與 **keep-set 聯集**（∪E(t)）兩處；folds、features、kNN、variants、fail-open、OOS cost gate 全部原封不動。實作：`sys.argv` 指到 `cv` 再 import `m56_percase_oracle`（頂層 script，import 即重跑原 CV = 免費 gate0 錨；副作用重 dump `results_M56_percase.json` 逐位同值）；`keep_set_e`/`model_pool_e`/`loo_J_e` 鏡射原函式、eq 參數化。

## 3. Gates（全綠）

- **gate0**：import 重跑重現原 CV —— Jstar 20/20 None、breaks 30+30、mean_pool 22.86、winner_hi/lo 與存檔 oracle 段 100/100×2 逐位相等。
- **selfcheck**：singleton-E 走新管線 —— fold0 四組 `loo_J_e` == 原值（None）、OOS **200×2 recs 逐位相同**（breaks 30==30）。
- **等價類 sanity**：winner ∈ E 全 assert 過；keep-set E 版 ⊇ singleton 版（OOS 全程 assert，0 違例）。

## 4. 主結果

**|E| 分佈**（E build ~2350 次官方 eval）：hi mean **1.39**（median 1、max 6、|E|>1 佔 **20/100**）；lo mean 1.27（|E|>1 佔 17/100）。最大類 |E|=6（cases 71/60/20）= `#22 #24 #25 #26 #27 #40`——全是 FC×FREE×PIN×WT×fs×W2.0 家族，彼此只差 FREE_ANCHORED/BND/MIB_ASPECT/FRAME_ASPECTS 疊層：**gated 旋鈕在無對應結構的案子上是恆等變換 → cost 逐位同值**（M51 clamp 同源機制）。等價類真實存在、非空想。

**E 化後 CV**：

| 量 | 原 M56 | M66 E 化 |
|---|---:|---:|
| train-LOO J\*（20 組） | 全 None | **仍全 None** |
| OOS strict-breaks | 30/200 | **26/200**（4 healed、**0 new**）|
| 最壞 break | case 96 @lo +9.09% | **不動**（+9.09%）|
| mean kept pool | 22.86 | 23.0（**mean chain pool 24.4**）|

兩 variant（knn / knn+band）逐位同結果（J=|train| 時 band 聯集 ⊂ kNN 全聯集）。healed 四筆 = case 37 @hi/@lo、case 38 @hi/@lo（其 winner 的 E 兄弟出現在 train 聯集）。

**Break 診斷**（`diag` 模式）：26 個殘留 break **全部 LABEL-ABSENT 且 |E|=1、0 個 HMIN-FLIP**——破口案的 winner 是 singleton 等價類（**連自己 chain pool 內都沒有 cost 等值替身**），且不在任何 train 案的等價類聯集；無一例是 subset-pool hmin 耦合誤選。

## 5. 判定與機制

**RED（OOS preservation < 100%，封卷維持）。** 依預註冊規則：無 green variant → 不進投影；且 mean kept pool 23.0 ≈ chain pool 24.4（最寬鬆可行模型已保留 94% 池），即使 0 breaks，RF 增益也已注定歸零（GREEN bar 0.05% 無從達到）。

機制升級：M56 的「winner 身分 case-idiosyncratic」精化為「**破口恰好集中在 winner 等價類為 singleton 的案子**」——等價類擴張只救得了 winner 有替身的案（4/30），而 26 個破口案的 winner 唯一、必須逐字命中 label 才能保 selection。方法論不對稱是真的，但與失敗根因**正交**：破口不在判準過嚴，在 label 結構（held-out winner 及其全部等值替身皆不在 train 集）。

## 6. 誠實範圍

- E 定義域 = 該案該 regime 的 chain_pool（winner 定義域）。全域被剪的 swap/OM 家族不在任何 E——但 model pool 本來就與 chain_pool 相交，此範圍選擇對 CV 零影響，只影響 |E| 統計的口徑。
- `diag` 只跑 knn variant（兩 variant break 集逐位相同、J=|train| 時二者 collapse）。
- 承 M55/M56 已知限制：audit_cache dt = n>100 K=12 counterfactual——本次未進投影，該限制未被觸及。
- 首跑被 harness 背景 10 分 timeout 於 E build 90/100 處殺掉 process tree（無 traceback、恰 +10:00）；cost cache 逐 10 案 flush → `Start-Process` detached 重跑斷點續完，結果不受影響。

## 復現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py -u m66_equiv_cv.py selfcheck   # gate0 + singleton-E 奇偶閘（全綠）
& $py -u m66_equiv_cv.py cv          # 主跑：E build + E 化 LOO/OOS -> RED（>10 分，用 Start-Process detached）
& $py -u m66_equiv_cv.py diag        # 26 break 全 LABEL-ABSENT |E|=1
```
