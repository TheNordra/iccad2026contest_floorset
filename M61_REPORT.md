# M61 — blocked exact-edge candidate → event frames（P4）：**RED**

日期：2026-07-17。Spec：`M57_PLAN.md` §6。執行模式：診斷先行（trace → event frame 合成 → FORCE_FRAME 實跑 → 官方 strict eval → proxy 仲裁）。
判定：**RED**——診斷前提為真（event frames 確實全是新 frame、96/97 開出新可行拓撲），但**分數價值恰為零**：6 案 97 個 event frames 最佳單案 weighted **+0.0001%**（case 65，且只是官方 cost 第 4 位小數的 tie 級改善），離 0.05% bar 差 500 倍。

## 一句話

`frame_candidates()` 的 obstacle-blind 是真的、「剛好越過 blocker 的 frame」也真的能解鎖被擋的 RIGHT/TOP exact-origin 候選並改變整個 pack——但 greedy 在放大的 outline 裡重排整案後，area/HPWL 稅恆 ≥ 解鎖那一個貼牆候選的收益：97 events 中 0 個達 bar、93 個持平或倒退、40 個直接崩到 shelf-fallback 級（cost 5.8-8.8）。「多開 frame 不如信 layout_score 選 4 個緊 frame」的 M25/M51 教義在 obstacle-aware 方向同樣成立。

## Kill gate 判定（spec 三擇一，中第三條）

1. **event 重複既有 frame？否**——97/97 個 event frames 對 FRM 全清單（該 host 的 `frame_candidates()` 完整輸出，含 max_trials 之外未試者）去重後全數存活（dup_dropped=0）：frame 生成器確實從不產生這些尺寸 → 前提成立、此殺不觸發。
2. **無新 feasible topology？否**——96/97 官方 feasible 且 positions ≠ winner run（唯一例外 case 62 H+1.685@153.2×204.3 收斂回同一佈局）→ 此殺不觸發。
3. **最佳單案 weighted <0.05%？是 → RED**——全 97 events 對 L1 live 基準（`results_L1_final.json` 逐案 cost）：最佳 **+0.0001%**（case 65）；realizable（proxy 會選）最佳同為 +0.0001%。其餘全部 ≤0。

## 數據（6 案、97 event frames、FORCE_FRAME 全實跑官方 strict eval）

| case | n | host k | frames | raw EVT | event frames | dup | fallback 級(cost>5) | proxy 選中 | 最佳事件（pct_l1，vs L1） |
|---|---|---|---|---|---|---|---|---|---|
| 62 | 83 | 1 | 25 | 17,638 | 18 | 0 | 10 | 0 | **+0.0000%**（H+1.685 → 同 cost 1.510741，positions 收斂相同）|
| 65 | 86 | 25 | 20 | 20,144 | 19 | 0 | 8 | 1 | **+0.0001%**（H+0.604@182.7² → 1.672924 vs 1.673243，proxy 會選）|
| 85 | 106 | 15 | 25 | 24,804 | 14 | 0 | 6 | 0 | −0.0399%（H+0.978@148.6×270.2 → 1.5307 vs 1.5096）|
| 88 | 109 | 27 | 20 | 8,715 | 12 | 0 | 4 | 0 | −0.0135% |
| 89 | 110 | 22 | 20 | 13,008 | 22 | 0 | 8 | 0 | −0.1668% |
| 99 | 120 | 41 | 20 | 344,542 | 12 | 0 | 4 | 2 | −0.0008%（H+0.057 → 1.308492 vs 1.308354，**proxy 會選但官方微退**）|

- **case 99 的 proxy mis-rank 警訊**：兩個 event（H+0.057 / W+0.029）proxy 均判優於全 84 池、officially 卻各 −0.0008%/−0.0017%——若把此類 frame 做成 live profile，wrapper 會選到微退化佈局。與「零收益」疊加 = 此軸不但無肉、還有下檔。
- **fallback 級 40/97 的機制**：EVT 是從 host 的**全部**嘗試 frame（含 pack 失敗、未計入 trials 者）收的；最小 base frames 本就 pack 不動，其 min-growth event 多半同樣 pack 失敗 → C++ `shelf_fallback` 佈局（feasible 但 cost 5.8-8.8）。診斷不再細分（不影響判定）。
- **改善量級參考**：case 85 曾在 winner host 自身 pre-LP cost（1.5725）上顯示 +0.08% 的假訊號——換成正確的 L1 live 基準（1.5096，pre-LP 池 proxy 實選產物）後為 −0.04%。**host 自身 pre-LP cost 不可當基準**（post-LP 重選常挑 pre-LP 較差的 host）。

## 誠實範圍

- **單 host**：每案只 trace winner host（M59/M60 同款 `winner_host`，anchor json cost 逐位 assert）；其他 83 隻 profile 的 event frames 未掃。但 winner host 已是該案最佳起點，其 event 全滅時其他 host 的 event 更難達 bar。
- **每 (base frame, 軸) 只取 min-d**（spec「每 base ≤2」＝ min ΔW + min ΔH）；更大的 d 未掃——但 d 越大 area 稅越重，方向不利。
- **d 的語意**：只保證越過「該候選被擋當下」的全部 blockers（`max(r.edge − m.origin)`，exact abutment 非 overlap 不加 MARGIN）；平移後可能撞新 blockers——由 FORCE_FRAME 實跑吸收，不是模型假設。
- **FORCE_FRAME 語意**：單 frame 走完整 per-frame 管線（REFINE/compaction/hpwl_push 全開、layout_score 不跨 frame 混選）；與「把 event frame 加進 frames 清單」的 ship 形不同（後者受 max_trials=4 排擠——event frame 面積必大於其 base，多半根本輪不到試），故本量測是 event frame 的**上界**語意。
- 儀裝涵蓋兩個 exact-origin 生成點：generic items loop（`item_candidates` 的 xv/yv，含 clamp 到 [0,xmax] 的 key 匹配）+ anchored first-pass（`adjacent_candidates_for_block` 的 `fw−w`/`fh−h`）；FREE_ASPECT single 路徑要求 boundary==0、無 exact origin，不涵蓋。EVT 中 src=A（anchored）僅 1 筆進入 event frames（case 65），其餘全為 src=I。
- trace env 採 `ICCAD_FRAME_EVENT_TRACE=<檔案路徑>`（鏡射 M60 檔案形，非 spec 字面 `=1`），stdout 完全不動以過 byte-gate。
- 純 quality 診斷，無 runtime 量測。

## 驗證鏈（全綠）

- byte-gate①：`constructive_m61.exe`（雙 env 皆關）positions 逐位 == `m53_l3_cache` run（6/6 案）。
- byte-gate②：trace on 時 stdout positions 仍逐位相同（6/6 案）。
- `winner_host` anchor cost assert：6/6 案逐位吻合 `results_L3_port_top32_area.json`。
- 官方 eval 一律 strict（`evaluate_solution` + `target_positions`）；proxy 仲裁用 `oc._proxy_metrics`（shapely vrel）併 84 池 cached pm 重算 hmin。

## 對 A6 的建議

- 死路 ledger 新增：**M61 event frames RED（拓撲真、分數零）**——obstacle-aware frame 生成方向封卷：event frame 解鎖 exact-origin 候選的收益恆被 outline 放大的 area/HPWL 稅吃掉（97 events 最佳 +0.0001%）；且 proxy 對此類微差 frame 有 mis-rank 下檔（case 99 兩例）。勿再嘗試「per-case 動態 frame」「blocker-aware frame 掃描」等變體；FRAME_ASPECTS/SCALES 靜態掃描已由 M51 條封。
- 與 M51 wide-CLAMP 的對照可寫：M51 的贏來自 clamp 下限把多個 aspect 收斂到**同一個**寬 outline（值不敏感、免 area 稅）；M61 證明反方向（微調尺寸繞 blocker）無肉——frame 軸的殘餘自由度兩側皆已探畢。

## 新增檔案

| 檔 | 說明 |
|---|---|
| `constructive_m61.cpp` / `.exe` | shipped 副本 + `ICCAD_FRAME_EVENT_TRACE=<file>`（FRM/EVT 記錄）+ `ICCAD_FORCE_FRAME=WxH`（強制單 frame）；雙 env 關 = 逐位 shipped |
| `m61_event_frame_probe.py` | driver（Phase A byte-gate+trace / B event 合成+去重 / C FORCE_FRAME+strict eval+proxy 仲裁 / D kill gate） |
| `m61_events.json` | 機讀結果（verdict / 6 案 97 events 全明細：d、base、cost、pct_l1、pct_anchor、proxy_selected） |
| `m61_cache.pkl` | run/gate/basecost cache（sig = 84-pool + m61 exe md5，斷點續跑） |
| trace 檔 | session scratchpad `m61_trace_<ci>.txt`（暫存，可重生） |

## 復現

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive_m61.exe constructive_m61.cpp
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m61_event_frame_probe.py          # 全 6 案（~4 分鐘）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m61_event_frame_probe.py --cases 85
# ICCAD_L1_POOL=1 + ADAPTIVE_POOL=0 由 driver 自設；基準 json：results_L1_final.json（live）/ results_L3_port_top32_area.json（anchor）
```

未動任何 shipped 檔 / CLAUDE.md / memory；未 commit；`m53_l3_cache.pkl` 唯讀。
