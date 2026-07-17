# M60 — anchored first-pass 牆面容量診斷（P3）：**RED**

日期：2026-07-17。Spec：`M57_PLAN.md` §5。執行模式：診斷先行、10 分鐘 kill gate。
判定：**立即 RED，不實作 beam**——假說的前件在資料上是空集（詳下），分叉掃描全程零觸發。

## 一句話

anchored first-pass（`constructive.cpp:630-688`）**從未**吃掉任何 boundary violator 的牆段：全 100 案 winner hosts 只有 2 個 anchored-cluster boundary violator（皆羽量案、且 pack 當下都在牆上）；9 個 traced 案的全部 26 個 movable violator，first-pass 結束時其需求牆段全數仍開放（`wall-open-after-fp`）——吃牆的是後面的 items loop（M27 packer frontier 域），不是 anchored greedy。self-fork = 0、upstream STRONG = 0、WEAK = 0、fp-implicated = 0。

## Kill gate 判定（spec 三重滿足，任一即 RED）

1. **Primary 範圍近乎真空**：全 100 案 winner hosts 中「最終落在 anchored cluster 的 movable boundary violator」僅 **2 個**——case 1（n=22，block 20，code L）、case 7（n=28，block 14，code T），皆最低權重帶（w=e^(n/12)≈6.3/10.3，vs 總權重 275418）。硬案 89/97/61/79/66 連 movable violator 都沒有（其 violator 全是 preplaced，屬 M57 域）。
2. **兩個 primary violator 都是 `pack-time-satisfied`**：first-pass commit 當下 selected bp=0（就在牆上），violation 是後處理（compaction/push/nudge）或官方 bbox 相對位移產生——與 first-pass 牆面容量無關，分叉掃描無從觸發。
3. **Secondary（資訊性擴充）全數排除 first-pass**：9 個 traced 案 26 個非 anchored movable violator，逐 bit 計算「PRE-only deficit vs first-pass 結束後 deficit」——**0 個 fp-implicated**；所有 violator 的 needed 長度 ≪ first-pass 後 max free gap（如 case 85：needed 12.3-24.2 vs gap 270.2/148.6；case 65：needed 9.3-28.0 vs gap 43-182.7）。牆段是在 items loop 才被消耗，或 violation 根本不是 frame-牆容量問題。

⇒ 「當時有更保牆的替代候選」的分叉**不存在**（更強：連「牆被 first-pass 吃掉」這個前件都不成立）→ RED，beam（B=4 lexicographic）不做。

## 數據

### Phase A 預掃（100 案，cache-only，零 exe 呼叫）

| 項 | 值 |
|---|---|
| movable boundary violators（官方 bbox 語意，EPS 1e-6） | **129 個 / 65 案** |
| 其中 anchored（mixed-cluster）成員 = **primary 範圍** | **2 個 / 2 案**（case 1 b20、case 7 b14） |
| 其中 pure-movable cluster 成員 | 大宗（ledger「123 cluster」的主體） |
| 硬案中完全無 movable violator | 89、97、61、79、66（violator 全 preplaced → M57 域） |

### Phase B/C traced（9 案 = 2 primary + 7 硬案 secondary；trace 全程 ~84s）

| case | n | host k | win pack | fp 成員數 | PRIMARY | SECONDARY（全部 `wall-open-after-fp`） |
|---|---|---|---|---|---|---|
| 7 | 28 | 16 | seq 9/52 | 3 | b14 `pack-time-satisfied` | 1 個（b21 needed 27.0 vs gap 100.0） |
| 1 | 22 | 38 | seq 83/477 | 2 | b20 `pack-time-satisfied` | 3 個 |
| 91 | 112 | 41 | seq 94/512 | 8 | — | 2 個 |
| 88 | 109 | 27 | seq 3/13 | 15 | — | 2 個 |
| 85 | 106 | 15 | seq 30/55 | **0** | — | 4 個（needed ≤24.2 vs gap ≥148.6） |
| 82 | 103 | 41 | seq 87/531 | **0** | — | 2 個 |
| 65 | 86 | 25 | seq 10/41 | 12 | — | 7 個 |
| 62 | 83 | 1 | seq 19/57 | **0** | — | 4 個 |
| 52 | 73 | 24 | seq 10/37 | 5 | — | 1 個（needed 21.0 vs gap 159.0） |

fp 成員數=0（85/82/62）= 該案根本沒有 mixed cluster → anchored first-pass 是 no-op，trivially 無嫌疑。

### 誠實範圍 / metric 定義

- **牆段數學**：對 violator 的每個 missing bit，strip 深度 = 其垂直向尺寸、needed = 沿牆尺寸；occupied = 與 strip 相交的已放 rects 投影 union；deficit = needed − max_free_gap（另記 total-free）。牆 = **frame 邊**（packer 語意）；violation 判定 = **官方 bbox 邊**（EPS 1e-6）——兩語意在 LEFT/BOTTOM 常重合、RIGHT/TOP 可有 slack，`pack-time-satisfied` 類即兩語意分歧的產物。
- **分叉定義**（本次零觸發，僅列存檔）：upstream fork = 只換單一 commit、state 凍結在 t' 當下的局部重算（不重模擬下游 cascade）；STRONG = alt 保住 gap ≥ needed 而 selected 使其 < needed；violator 維持 selected 尺寸（不聯合 reshape）；corner（雙 bit）逐 bit 獨立判。
- **Secondary 的侷限**：非 anchored violator 的自身 commit 在 items loop、不在 trace 內——只回答「first-pass 單獨是否已毀其牆」（答案：全數否）。items loop 內部的吃牆行為屬 M27 packer 條目管轄，非本 probe 範圍。
- trace env 採 `ICCAD_ANCHOR_TRACE=<檔案路徑>`（鏡射 M59 `ICCAD_REFINE_DUMP` 檔案形），非 spec 字面的 `=1`——讓 stdout 完全不動以過 byte-gate。
- winner host = M59 同款 `winner_host`（top-32 pre-LP proxy + post-LP proxy 重選、對 anchor json cost 逐位 assert）；violator 判在 host 的 **placer 輸出**（`("run", ci, ksel)`，pre-LP）。

### 驗證鏈（全綠）

- byte-gate①：`constructive_m60.exe`（trace off）positions 逐位 == `m53_l3_cache` run（9/9 案）。
- byte-gate②：trace on 時 stdout 仍逐位相同（9/9 案）。
- replay 自檢：WIN pack 全部 commit rects 兩兩無 overlap（TOL 1e-6）。
- winner_host anchor cost assert：100/100 案逐位吻合 `results_L3_port_top32_area.json`。

## 對 A6 的建議

- 死路 ledger 新增：**M60 anchored first-pass 牆面容量 RED（前件空集）**——anchored-cluster violator 全集 2 個且 pack 當下皆在牆上；first-pass 對其餘 26 個 movable violator 的牆全數保持開放；beam / 保牆排序 / lookahead 任何變體都不必做。residual 吃牆行為在 items loop → 歸 M27 packer 條目（已封）。
- plan §7 的 M26 oracle-perm 加註可寫：「anchored first-pass 內部順序已由 M60 診斷——不是排序問題，first-pass 根本不產生 boundary 容量衝突」。
- Case 1/7 的 `pack-time-satisfied` 機制（後處理把牆上 block 拉離官方 bbox 邊）若要追，屬後處理/邊界 nudge 軸，非 M60 範圍；兩案權重合計 ~0.006% 級，不值得。

## 新增檔案

| 檔 | 說明 |
|---|---|
| `constructive_m60.cpp` / `.exe` | shipped 副本 + `ICCAD_ANCHOR_TRACE=<file>` 儀裝（PACK/PRE/MEM/CAND/WIN 五種記錄；trace off 逐位 = shipped） |
| `m60_anchored_deficit.py` | driver（Phase A 預掃 / B trace+byte-gate / C deficit 回推+分叉掃描） |
| `m60_forks.json` | 機讀結果（verdict / 預掃 129 violators 全清單 / 9 案 traced 細節） |
| `m60_cache.pkl` | gate/trace 完成旗標（sig = 84-pool + m60 exe md5） |
| trace 檔 | session scratchpad `m60_trace_<ci>.txt`（暫存，可重生） |

## 復現

```powershell
& "C:\msys64\ucrt64\bin\g++.exe" -O3 -std=c++17 -o constructive_m60.exe constructive_m60.cpp
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m60_anchored_deficit.py            # 全流程
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m60_anchored_deficit.py --prescan-only
# --cases 7,1 / --max N 可縮範圍；ICCAD_L1_POOL=1 + ADAPTIVE_POOL=0 由 driver 自設
```

未動任何 shipped 檔 / CLAUDE.md / memory；未 commit；`m53_l3_cache.pkl` 唯讀。
