# M65：L1 池遺漏 K24 cell 補量 — RED（pilot gate，0 movers）

**日期**：2026-07-19　**工具**：`m65_l1_cell.py` + `m65_cache.pkl`（sig=profiles+exe md5）+ `results_M65_l1cell.json`
**定位**：offline 錨專用探測，永不 ship；全程未動任何 shipped 檔案、未重編 exe。

## 1. 前提（code 已驗證）

`optimizer_constructive.py:376-379`：`_L1_BASE = list(_PROFILES)` 快照在 `_PROFILES.extend(_L1_EXTRA)` **之前** → `ICCAD_REFINE_ITERS=24` tier 只複製 41 隻 base。84 池 = 41 base + 2 extras + 41 K24；兩隻 `_L1_EXTRA`（OS16×free 家族：os16_fc_anchored_bnd_pin_tight、os16_fc_pin_tight）沒有 K24 版本。gate0 以 assert 寫死此事實（池=84、兩 K24 combo 不在池中、base extras 在池中，全 OK）。

## 2. 方法

- 兩隻候選 cell = `_L1_EXTRA[j]` + env 疊加 `ICCAD_REFINE_ITERS=24`（j=0,1）；subprocess 直呼 shipped `constructive.exe`，env 先剝除父行程全部 `ICCAD_*`（regression_suite 衛生），timeout 600s，ThreadPool 11 workers、大 n 先排。
- 官方仲裁：`m53_l3_probe.cost_eval`（strict `evaluate_solution`，target_positions 傳入）。
- gate0 anchor sanity：案 0/85/99 `cost_eval(anchor positions)` 逐位 == `results_L1_final.json` cost（全 OK）。
- 預註冊 gate：pilot 15 重案（85-99）mover = 官方 cost < anchor − 1e-6；**0 mover → RED 停止**（不跑 full / l3seed）。

## 3. Pilot 結果（15 重案，30 runs，95s wall）

| case | n | anchor | os16_ab_K24 (Δ) | os16_fc_K24 (Δ) |
|-----:|----:|---------:|------------------:|------------------:|
| 85 | 106 | 1.509601 | 1.667623 (−0.158) | 1.667623 (−0.158) |
| 86 | 107 | 1.294786 | **1.294786 (tie)** | 1.413023 (−0.118) |
| 87 | 108 | 1.280293 | 1.516227 (−0.236) | 1.516227 (−0.236) |
| 88 | 109 | 1.385184 | 1.549704 (−0.165) | 1.411910 (−0.027) |
| 89 | 110 | 1.523183 | 1.565925 (−0.043) | 1.553789 (−0.031) |
| 90 | 111 | 1.325592 | 1.347618 (−0.022) | **1.325592 (tie)** |
| 91 | 112 | 1.325506 | 1.361105 (−0.036) | 1.401162 (−0.076) |
| 92 | 113 | 1.246188 | 1.345985 (−0.100) | 1.248808 (−0.003) |
| 93 | 114 | 1.299859 | 1.657466 (−0.358) | 1.548906 (−0.249) |
| 94 | 115 | 1.311070 | 1.427544 (−0.116) | 1.427544 (−0.116) |
| 95 | 116 | 1.175079 | 1.316806 (−0.142) | **1.175079 (tie)** |
| 96 | 117 | 1.295205 | 1.373630 (−0.078) | 1.373630 (−0.078) |
| 97 | 118 | 1.187113 | **1.187113 (tie)** | 1.226594 (−0.039) |
| 98 | 119 | 1.278123 | 1.302990 (−0.025) | 1.299749 (−0.022) |
| 99 | 120 | 1.308354 | 1.352080 (−0.044) | 1.334839 (−0.026) |

（Δ = anchor − cost；負 = 退步。26/30 runs 嚴格退步、4 tie、0 改善。）

## 4. 判定與機制

**RED（pilot gate：0 movers → 停止）。兩隻 K24 cell 不加、84 池不動、L1 錨 1.3176 / L3 鏈 1.2978 全不動。**

機制（cache 驗證）：4 個 tie（86/97←ab、90/95←fc）的 K24 輸出 positions 與錨**逐位相同**——這四案的錨 winner 正是對應 K12 extra，REFINE 12→24 在其勝出案上是 **exact no-op**（refine 已在 pass<12 收斂）；其餘 26 runs 全數付 REFINE-up 毒（layout_score≠true cost 非單調，M53 L1 「全域 override +0.105%」同機制，此處逐案顯形至 −0.36）。即：**K24 疊在 OS16 extras 上，贏處是 no-op、輸處是毒**——遺漏的兩 cell 對 heavy band 期望值恰 0。

## 5. 誠實範圍

- Pilot 限 15 重案（n=106-120）。extras 的 L1 贏案有數個在 mid band（80/73/82/78/75/66、64/35）未量測——但 gate 預註冊於重案帶，且 in-band 四個 extras 贏案已證 K24 全為逐位 no-op，mid band 同族收斂性質期望一致；權重端 mid 案貢獻遠低。
- full（oracle-min 0.05% bar）與 l3seed（LP 種子 0.10% bar）依 gate 未動用；工具已備妥（`full`/`l3seed`/`report` 模式、86-pool proxy 重選 + `m53_l3_cache.pkl` sig 驗證、L3 弱基準保守邏輯），若日後要重開直接跑。

## 6. 附帶修復（本次真正有價值的產出）

**commit 25f3eb0（"remove stale proxy debug scripts"）誤刪 `proxy_analysis.py`**——它是 27 隻工具的 load-bearing 依賴（`rf_score_model.py`、`analyze_constructive.py`、`profile_vs_portfolio.py`、`m53_l3_probe.py` portfull、m55/m56/m57-m61 probes…），**整條 `regression_suite.py` 送件前 gate 鏈因此斷裂**；M62-M64 全是 cache-only 純分析才沒踩到。已自 `25f3eb0^` bit-exact 還原並隨本 commit 入庫。同 commit 刪的 `proxy_dbg.py` 確為 stale（零依賴），維持刪除。

## 復現

```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m65_l1_cell.py gate0   # 全綠
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m65_l1_cell.py pilot   # 0 movers -> RED
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m65_l1_cell.py report  # dump json
```
