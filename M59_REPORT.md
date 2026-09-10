# M59 REPORT — REFINE rejected states → L3 LP 種子 pilot（P2）

**日期**：2026-07-16　**判定**：🔴 **RED（weighted gain = +0.0055% < 0.05% bar，且無顯著單案）**　**shipped 檔零變更、無 commit；第二階段 dual-guided relation flip 依 kill gate 一併不做**

## 一句話結論

REFINE 逐 pass 丟棄的 `c2` 拓撲州**確實全是新拓撲**（251/251 dumped states 的 pair-relation signature 互異、且無一與錨拓撲或 host pre-LP 輸出拓撲重合），但**品質盆地太遠**：39 個 rejected-c2 種子過 LP 後 **0 個**打贏該案錨值；唯一 mover 是 case 88 的**替代 frame pre-refine c1**（f0 initial pack，非本 probe 假說標的），LP 後 1.325391→**1.323154**（單案 −0.17%、weighted **+0.0055%**）——離 bar 9 倍。「pre-LP 差、post-LP 贏」的 L2-seeds 機制**不轉移**到 mid-pipeline 州。

## 機制解釋（為何 L2-seeds GREEN、M59 RED）

`c1`/`c2` 是 **pre-compaction/pre-push 中間態**；凍結拓撲 LP 只做平移，代替不了 compaction 的拓撲改寫（方向性重排 + void 擠出）。L2-seeds 之所以 GREEN，是其種子為**全管線輸出**（compact+push 已跑完、僅上游權重 jitter）——拓撲已是「compaction 後」的形，LP 只需微調平移。M59 種子少了這一段，post-LP 仍掛著 initial-pack 的 area/HPWL 結構：6/7 案 best post-LP 高於錨 **+0.06%～+13.6%**（中位 ~+3%）。近錨的例外是 case 65 的 c2 種子 f0r0（1.598433 vs 錨 1.597457，僅 +0.061%）與 case 88 的 c1 直接翻贏——顯示殘值在「**替代 frame 的早期州**」而非「refine 拒絕州」，與 M61（event frames、frame 軸）相鄰，與 refine 軸無關。

## 量測鏈（全綠）

| 步驟 | 結果 |
|---|---|
| winner host 查詢（鏡射 `mode_portfull` top-32 + post-LP proxy 重選，`m53_l3_cache.pkl` 唯讀） | 7/7 案 ksel 的 cache LP cost == 錨 json cost **逐位相同**（assert） |
| LP 模組 sanity（gate0 語意：strict eval 錨 positions） | 7/7 案 cost **逐位重現**錨 json（assert） |
| byte-gate（`constructive_m59.exe` dump-OFF vs cache `("run",ci,ksel)`） | **7/7 逐位相同 PASS**（m46 副本未擾動） |
| dump run stdout（dump-ON vs dump-OFF） | **7/7 逐位相同**（儀裝零行為變化） |
| dump 州數 / distinct signature | 251 states / **251 distinct**（exclusion 命中 0：錨與 pre-LP run 拓撲皆未在中間態出現） |
| 種子 LP（cap 8/案、`--area`、2 passes、官方 strict keep-guard） | 55 seeds（16 c1 + 39 c2）全數 LP 完成、post-LP 全 feasible |
| `m53_diff_results.py` 複核 | 1 mover（case 88 improved）、0 regressed、100/100 feasible、total −0.0055% 一致 |

## Per-case 結果（錨 = `results_L3_port_top32_area.json`，total 1.3003478581）

| case | n | host k | states | seeds | best post-LP（種子） | 錨 | d | wContr |
|---|---|---|---|---|---|---|---|---|
| 62 | 83 | 1 | 52 | 8 | 1.642264（f1r9, c2） | 1.445165 | 0 | 0 |
| 65 | 86 | 25 | 30 | 8 | 1.598433（f0r0, c2） | 1.597457 | 0 | 0 |
| 85 | 106 | 15 | 52 | 8 | 1.472983（f2r-1, c1） | 1.458195 | 0 | 0 |
| **88** | **109** | **27** | **7** | **7** | **1.323154（f0r-1, c1）** | **1.325391** | **+0.002236** | **+0.0055%** |
| 89 | 110 | 22 | 52 | 8 | 1.632489（f3r8, c2） | 1.490417 | 0 | 0 |
| 91 | 112 | 41 | 27 | 8 | 1.335561（f0r-1, c1） | 1.296344 | 0 | 0 |
| 97 | 118 | 41 | 31 | 8 | 1.217144（f0r-1, c1） | 1.182295 | 0 | 0 |

合成 total（僅 case 88 覆寫）：**1.3002763410**（−0.0055% vs 錨）。注意 4/7 案的 best 種子是 **c1（pre-refine）**而非 c2——rejected-c2 假說本體 0 hit。

## 唯一 mover：case 88（n=109，f0 pre-refine c1 + LP）

host #27 的 frame 0 initial pack（尚未 refine/compact/push）strict cost 1.390279，LP 兩 pass 後 **1.323154**，勝過錨（frame 選擇 + 全後處理 + LP）的 1.325391。單案降幅 0.17% < 0.3% 顯著門檻、wContr +0.0055% < 0.01% → 不構成「顯著單案」。附帶觀察：case 88 只 dump 到 7 州（f0/f1/f2 首個 refine pass 的 guided re-pack 失敗即 break，僅 f3 有 c2）——refine 在此 host 幾乎不產出可用州，殘值明確在 frame 軸。

## Kill gate 判定

- **weighted +0.0055% < 0.05%** → 成立。
- **無顯著單案**（唯一 mover 單案 −0.17% < 0.3%、wContr +0.0055% < 0.01%）→ 成立。
- ⇒ **RED**。依 spec，第二階段（|dual| top active separation rows × 其餘 3 relations、beam 4）**一併不做**。
- 撿回路徑（spec 註明「訊號貼近 bar 才考慮」的 postproc-stdin 回灌 compact/push 再 LP）：+0.0055% 離 bar 9 倍，**不啟動**。若未來重開，方向應是 M61 frame 軸（本 probe 的 c1 訊號指向 frame 選擇，非 refine 州）。

## 復現

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive_m59.exe constructive_m59.cpp
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m59_refine_seed_probe.py          # 7 案全跑 ~3.5 分（cache 命中則秒回）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m59_refine_seed_probe.py --cases 88   # 單案
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m53_diff_results.py results_L3_port_top32_area.json results_M59_refine_seed.json
```

註：`m53_l3_cache.pkl` 的 lp 鍵是 `("lp", ci, k, 8, True)`（portfull 當時 `--iters 8` 上限跑法）；本 probe 的種子 LP 依 spec 用 2 passes。

## 新增檔案（shipped 檔與既有 probe 一律未動）

- `constructive_m59.cpp` / `constructive_m59.exe` — `constructive_m46.cpp` 副本 + `ICCAD_REFINE_DUMP=<path>`（逐 frame pre-refine c1（r=-1）+ 逐 pass c2，`%.17g`；gate off 零行為變化，dump run stdout 已驗逐位不變）
- `m59_refine_seed_probe.py` — driver + LP 代碼整份複製自 `m53_l3_probe.py`（未 import/patch 原檔）
- `m59_cache.pkl` — byte-gate + 種子 LP cache（sig = pool + md5(m59 exe)，斷點續跑）
- `results_M59_refine_seed.json` — 錨複本、case 88 覆寫（`m53_diff_results.py` 可直接 A/B）
- `m59_rows.json` — 逐案機器可讀表
- `M59_REPORT.md` — 本報告
- dump 檔（`m59_dump_{ci}.txt`）在 session scratchpad，不留 repo
