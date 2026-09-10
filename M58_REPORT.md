# M58 REPORT — compute_nsoft 官方分母修正 probe（P5）

**日期**：2026-07-16　**判定**：🔴 **RED（weighted |delta| = 0.0001% < 0.05% bar）**　**shipped 檔零變更、無 commit**

## 一句話結論

C++ `compute_nsoft` 的 MIB 分母 spec 落差（distinct-shapes−1 vs 官方 group_size−1）**是真實且 live 的**——修正後 53/2700 profile runs 的 positions 改變、29/100 案被觸及、且 case 6 因此翻 winner 拿到 −3.31% 的單案改善——但影響幾乎全落在**低權重小案**（n≤60 佔 22/29、n>100 恰 0 隻），weighted delta 僅 **−0.0001%**（官方分母方向微贏），離 bar 500 倍。**csc 對分母敏感、但分數不敏感 → RED，不 ship。**

## 機制解釋（為何小案集中、大案零觸及）

csc = `(area + hw·hpwl)·exp(2·(bv+gf)/nsoft)`。官方分母把 nsoft 加大 +2~+6（每案恰 1 個 MIB group、group_size−1 = 2..6，全 100 案）。這對**小案是大幅相對變化**（小 nsoft 基底 → violation 懲罰顯著鬆綁 → compaction 候選排序翻轉），對**大案是雜訊**（nsoft 基底大、+2~+6 攤薄 → 排序不動，n>100 觸及 0 隻）。而評分權重 e^(n/12) 恰好反向：會動的案權重最低 → 分數上打平。

## 量測鏈（全綠）

| 步驟 | 結果 |
|---|---|
| gate0 byte-gate（m58 exe flag-off vs shipped exe，2 案×6 kept profiles） | **12/12 逐位相同 PASS** |
| driver 忠實度 sanity（side-0 全池 proxy 選擇 vs `results_shipped_m51.json`） | **positions 100/100 逐位相同、cost 100/100 <1e-12** |
| 全池 diff（100 案 × shipped-form kept pool × 兩 env，5400 runs） | 53/2700 profile runs 有 position 變化、29/100 案觸及 |
| **winner-profile diff（spec 原判準）** | **0/100**——現行 winner profile 的 positions 全數不變 |
| 兩側 proxy 選擇 + strict eval | 1 個 selection flip、1 個 cost mover（case 6） |
| weighted total | side0 **1.3264731049**（=shipped）→ side1 **1.3264712033**，delta **−0.0001%** |

註：spec 原判準（winner-profile diff）單獨看即是 RED（0/100）；本 probe 加跑全池兩側，額外抓到「非 winner profile 改變 → proxy 翻 winner」的二階路徑（case 6），結論仍 RED。

## 唯一 mover：case 6（n=27，權重 9.49 / 總權重 275418 ≈ 0.0034%）

| | side0（shipped 分母） | side1（官方分母） |
|---|---|---|
| winner profile | #13 | **#4**（其 positions 在官方分母下改變） |
| strict cost | 1.6676133220 | **1.6124129943**（−3.31%） |
| feasible | ✓ | ✓ |
| dW（加權貢獻） | — | −1.902e-06 |

單案品質增益是真的（compaction 在官方分母下少怕一次 violation、選了更小 bbox 的候選），但 n=27 權重太低，撐不起任何 weighted 量級。

## 觸及案分佈（position 有變的 29 案）

- **band**：n≤60 → 22 案；60<n≤100 → 7 案（63/65/69/78/83/93/100）；**n>100 → 0 案**。
- 逐案（idx: 變動 profile 數）：0:9、1:3、2:7、4:1、5:1、6:4、7:1、8:1、11:1、13:3、14:1、15:1、16:2、19:1、20:1、22:1、26:2、27:1、29:2、30:1、33:1、36:1、42:1、44:1、48:1、57:1、62:1、72:1、79:1。
- official nsoft 增量：全 100 案各恰 1 個 MIB group，增量 +2~+6（`m58_nsoft_probe.py` dataset 掃描；per-case 明細見 probe 輸出）。

## 影響面稽核（spec 步驟 1，程式碼錨點）

| 使用點 | 判定 |
|---|---|
| `compute_nsoft`（`constructive.cpp:1033-1043`）→ `compact_layout` csc（`:1065`） | **唯一 live 使用點**（本 probe 量測對象） |
| `csc_of` 註解提及 `ICCAD_FRAME_CSC`（`:1049`） | **無對應程式碼**——僅註解殘留，grep 全檔無 gate |
| METRICS stderr（`:1631-1640`，自行另算 nsoft） | **cosmetic**——wrapper 不解析 METRICS（proxy 用 shapely，`optimizer_constructive.py:673-705`） |
| Python proxy nsoft（`optimizer_constructive.py:691,701`） | **已是官方 group_size−1 語意**，無落差 |

## Kill gate 判定

- ~~無 position 變化~~ → 不成立（53 runs 有變；csc 確實對分母敏感）。
- **weighted |delta| < 0.05%** → **成立**（0.0001%）→ **RED**。
- 附帶：即便未來想撿 case 6 的 −3.31%，也只值 −0.0001% weighted，且 ship 要付 M51 級重驗鏈（re-audit + `rf_score_model.py` regen + m49 三 gate + 官方 eval）——完全不划算。**此軸封卷**：spec 落差存在但無分數後果；勿再以「分母不對」為由重開。

## 復現

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive_m58.exe constructive_m58.cpp
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m58_nsoft_probe.py all   # gate0 + diff + eval，~6 分鐘（cache 命中則秒回）
```

## 新增檔案（shipped 檔一律未動）

- `constructive_m58.cpp` / `constructive_m58.exe` — shipped 副本 + `ICCAD_NSOFT_OFFICIAL` flag（僅 `compute_nsoft` MIB 項 + main env 解析兩處；cluster/boundary 項與 METRICS 未動）
- `m58_nsoft_probe.py` — driver（gate0 / diff / eval / all；cache sig=(pool, md5(exe))）
- `m58_cache.pkl` — run/proxy/cost cache
- `results_M58_nsoft.json` — 逐案兩側 cost、diff 明細、verdict
- `M58_REPORT.md` — 本報告
