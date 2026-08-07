# M62 — LP in-window overhead break-even（cache-only 純分析）：**GREEN（超標）——α\*=1.0，M54 RED 是 cap 網格假影**

日期：2026-07-18。執行模式：純分析（`m62_break_even.py` import `m54_lp_rf_model`，資料全來自 `m54_cache.pkl`；不跑新 LP、不跑官方 eval、不碰任何 shipped 檔）。
判定：**GREEN，且超出原題設**——原問「build+guard 要加速多少（α\*）才有 strict weak-win」，答案是 **α\*=1.0（零加速）在全部 β 檔位皆成立**：M54 的「任何 gate 組合皆無 strict weak-win」只在其掃過的 cap≥0.5s 網格內為真；cap∈{0.1, 0.25}s 的 **skip** gate 在 α=β=1（現行 Python 原速）就已全 20 cells 弱贏。break-even 問題消解；殘餘風險從「速度」移到「skip 預測器可部署性」與「grader 機速未知」。依 spec，GREEN = 只交報告與工程量估算，**不動工**。

## 一句話

M54 在 M=6-8 的 floor-cliff 稅是 `0.3·dt/t` 級、隨 dt 線性——把 dt 壓到 0.1-0.25s 量級，稅就掉到逐案 quality 增益之下；M54 的 cap 網格（最小 0.5s）恰好停在毒性線上方，M62 往下掃一格就翻盤。贏不需要 C++：需要的是「跑之前就知道這案 LP 要多久」（skip 語意 = oracle 時間預測，abort-only 免預測器 gate 的 wins = **0**）。

## Sanity 鏈（四項全綠，session 有效）

1. anchor total（RF=1.0）**1.3264731049** 重現（assert 過，= `results_shipped_m51.json`）。
2. α=1, β=1、**M54 原 CAPS（≥0.5s）**：clear weak-win = **0**——逐字重現 M54 結論（closest = `n>110 cap=0.5 it2 skip` 恰 +0.000/+0.000 的退化 gate，即 ledger 說的「LP 永不觸發」）。
3. α=1, β=0（M54 β=0 kill test）：最佳 gate 最壞 cell **+0.480%**——與 ledger「+0.48%」逐字吻合。
4. α=0, β=0：`n>0 cap=- it2` 全 cells [−0.909%, −0.888%]；RF=1 品質 1.326473→1.314658（**+0.8907%**，= M54 的 +0.891%）。

## 主結果

**WIN grid（eps=0、win_min=0.01%）：48/48 個 (α,β) 格全部存在 clear weak-win gate**；frontier 對每個 β∈{1, 0.5, 0.2, 0.1, 0.05, 0} 皆 α\*=1.0。α=1, β=1（零加速）的全部 6 個 win gates——**全是 skip、abort-only = 0**：

| gate（皆 skip） | worst cell | best cell | RF=1 品質增益 |
|---|---|---|---|
| `n>0 cap=0.25 it1` | −0.014% | **−0.356%** | **+0.3563%**（跑 76 案、75 案改善）|
| `n>60 cap=0.25 it1` | −0.003% | −0.345% | — |
| `n>0 cap=0.1 it1` | **−0.033%** | −0.060% | +0.0604%（跑 55 案、54 案改善）|
| `n>60 cap=0.1 it1` | −0.024% | −0.051% | — |
| `n>0 cap=0.1 it2` | −0.012% | −0.016% | — |
| `n>60 cap=0.1 it2` | −0.009% | −0.012% | — |

cell matrix（cap=0.25 it1 skip @α=β=1）：thin 端全在 **M=6**（−0.014~−0.100%，floor-cliff 機制殘留但已壓平），M≥8 全域 −0.2~−0.36%。主要貢獻集中：case 88（1.3852→1.3259，dt=0.239s）、79、69、65、75、68、84、54、89、67——**case 88 一案 ≈ +0.14% 佔增益 4 成，且 dt=0.239 離 cap 0.25 僅 4% margin**（見穩健性）。

加速的價值（非必要、但擴大戰果）：β=0.2（5× solver）時最佳 gate 增益 −0.399%、β=0.1 → −0.694%——加速的作用是讓更多 +0.891% 天花板塞進安全 cap 內，不是解鎖 win。

## 穩健性（判定的實質內容）

- **機速掃描**（grader 速度未知；全部 pass 時間 ×s、gate 固定）：`cap=0.1 it1 skip` 在 **s∈[0.5, 1.5] 全 W**（worst −0.106~−0.012%）、s=2 掉到 +0.000（weak 不 clear）、s=3 w；`cap=0.25 it1` 只活 s∈{0.7, 1.0}；**s≥2 時 48 gate 全滅（#win=0）**。⇒ 此 win **不是 machine-speed-independent**——與 M41-M50「median/機速無關才 ship」doctrine 的明確偏離，需要機速校準（見工程量）。
- **逐案 timing jitter**（每案 ×U[0.7,1.3]、40 draws；同時模擬預測器誤差）：`cap=0.1 it1` **40/40 全勝**（worst-cell 範圍 [−0.080%, −0.005%]）；`cap=0.25 it1` 27/40（[−0.168%, +0.159%]）。不對稱機制：jitter 把案擠出 cap = 只損增益（skip 免費）；擠進 cap = 可能引毒——cap=0.1 的候選集毒性有界、cap=0.25 邊界上掛著 case 88。
- **eps 敏感度**（每 pass +2ms/+5ms additive floor）：frontier 不動（全 α\*=1.0）。
- **win_min 敏感度**：cap=0.25 worst −0.014% 在 win_min=0.02% 即失格；cap=0.1 worst −0.033% 撐到 0.03%。兩 gate 的「厚利」都在 M≥8 cells。

## per-case dt 預算（0.3·dt/t 稅的量化；表節錄）

n≥110 重案在 M=6@12c 的 floor 免費額度全為 0、dt_tax（above-floor 打平預算 @12c）僅 7-129ms（case 99: 26ms、93: 129ms、88: 242ms）——**這就是 M54 all-gate 死因**：重案 LP chain 0.4-13.8s ≫ 預算兩個數量級。micro-cap skip 的本質 = 把這些案全部 0 成本跳過，只收「chain ≤0.1-0.25s 且有 proxy-guard 增益」的 55-76 案。safe-set 參考（逐案在全 20 cells 淨≤0 的保證集）@α=β=1 = 27 案 / −0.109%——aggregate cap-gate（−0.356%）超過它，因 cell 內允許跨案淨額。

## 誠實範圍

- **skip = oracle 時間預測**：模型用「真實 chain 時間 vs cap」判 skip。可部署形必須在 **build 之前**以案特徵（n、edge 數、LP 尺寸）預測——「先 build 再決定」不可行（全案付 build+guard = β=0 kill test 的 +0.48% 敗形）。預測器容錯已由 jitter 量測近似（±30% → cap=0.1 40/40）。此預測是 size→time 回歸，**與 M56 封殺的 winner-identity 預測不同類**（連續、結構性、有 graceful failure），但仍未建、未驗。
- **機速非獨立**（上節）：贏帶 s∈[0.5,1.5]；本機 vs grader 的絕對速度差未知。可用 wrapper 已有的 per-profile 計時做**相對 cap 校準**（cap 以「參考機秒」計價），把機速軸轉成 jitter 軸——工程假設，未驗。
- **薄 margin cells**：worst cells（M=6）−0.014~−0.033% 與模型誤差同量級（T 的 cores-scaling 用 audit-cache 比值、dt 為本機單次量測）；厚利在 M≥8。
- α 單一係數同縮 build+guard（同為 Python、同批 C++ 化）；β 乘 t_solve。**scipy linprog 底層已是 HiGHS C++** → β≪1 不是移植可得（warm-start / x-y 解耦另計）——但本結論在 α=β=1 已成立，此注意事項只影響「擴大戰果」檔位。
- 加速 cells 假設 C++/向量化版逐位重現 scipy 鏈的 keep 決策與品質（只縮時間）；α=β=1 主結論不依賴此假設（全為實測值）。
- q 路徑 = cache 的 proxy-guard keep + 官方 cost_after（M54 dep 已證 100/100 feasible、0 回歸）；iters=1 形即 cache 鏈第一 pass，語意精確。
- M 網格 {6,8,11,14,20}、cores {4,8,12,16} 沿用 M54；官方 median 仍未知。

## 對 CLAUDE.md / 下一步的建議

- **M54 ledger 條更正範圍**：「任何 band×cap×iters×skip 組合皆無 strict weak-win」→ 加註「限 cap≥0.5s 網格；M62 證 cap 0.1-0.25s skip gate 在零加速下弱贏」。復活條件從「median ≥11s 或 RF=1.0」擴為三路，新增 **M62 micro-cap skip 路徑**（條件：時間預測器 + 機速校準兩 probe 過關）。
- **C++ / 向量化 build+guard 軸：moot**（不需做即有 win；做了只擴大）。β 軸（solver 加速）同 moot。
- **不動工**（spec 明定 GREEN 行為 + 使用者「送件檔位不動」方針）。若未來要 ship，需依序：(1) pre-build chain-time 預測器（特徵 = n / b2b·p2b 邊數 / LP 行列數，100 案 LOO 驗證，容錯目標 ±30%）~1 天；(2) wrapper 相對機速校準（cap 計價於參考機秒）~0.5 天；(3) 預測器-in-the-loop 重跑本模型 + `regression_suite.py` 全 gate + `m48_coldstart_dryrun.py`；(4) env-gated ship 形（預設 off）。推薦檔位：**`cap≈0.1s、iters=1、skip` 為主**（穩健：40/40 jitter、s∈[0.5,1.5]）、cap=0.25 為 upside 檔（需校準後再議）。

## 檔案

| 檔 | 說明 |
|---|---|
| `m62_break_even.py` | break-even 模型（import `m54_lp_rf_model` 復用 load_walls/cache/常數；α×β×gate 掃描 + sanity 四項 + detail：win 清單/組成/機速/jitter + dt 預算 + safe-set + eps 敏感度）|
| `m62_stdout.txt` | 本次完整輸出（未 commit，可重生）|

## 復現

```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m62_break_even.py            # 全套 ~6 分鐘（含 dataset 載入）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m62_break_even.py --win-min 0.03   # 門檻敏感度
# 依賴：m54_cache.pkl（sig=shipped_m51/iters2/area）、audit_cache.pkl、results_shipped_m51.json
```

未動任何 shipped 檔；`m54_lp_rf_model.py` 未修改（import 復用）。
