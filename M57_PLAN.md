# M57–M61 平行 probe 計劃 + A6 彙整（2026-07-16 定案）

來源：外部 codex gap 報告（2026-07-16），行號級主張已於本日逐行驗證後定案。
執行模式：M57–M61 各自獨立 session **非同步平行**執行；A6 彙整 session **最後單獨**跑。

---

## 0. 已驗證的 code 事實（各 probe 的立足點，勿重驗、勿被 ledger 勸退）

| 主張 | 驗證結果 | anchor |
|---|---|---|
| L3 repair 對 frozen violator 連 LP 都沒建 | ✅ `return None` | `m53_l3_probe.py:278-281` |
| bbox 變數 bounds 已自動擋「更外側 frozen block」 | ✅ infeasible 自動成立 | `m53_l3_probe.py:324-333` |
| tied 等式讓已滿足 boundary blocks 跟邊同動 | ✅ | `m53_l3_probe.py:255-273` |
| 官方 N_soft 的 MIB 項 = group_size−1 | ✅ | `iccad2026_evaluate.py:461-465` |
| C++ `compute_nsoft` MIB 項 = distinct-shapes−1（MIB 統一後恆=0）| ✅ 真 spec 落差 | `constructive.cpp:1038,1041` |
| Python proxy nsoft 正確 → live 影響僅 C++ compaction csc | ✅ | `optimizer_constructive.py:698-701` |
| REFINE rejected `c2` 只當 guide 即丟 | ✅ | `constructive.cpp:1561-1569` |
| L3 每 pair 只留單一 max-gap disjunct（表達不了 relation flip）| ✅ | `m53_l3_probe.py:213-231` |
| M26 oracle-perm 只測 yx/xy 兩 scalar key、碰不到 anchored first-pass 內部排序 | ✅ | `oracle_perm_probe.py:39-40` + `constructive.cpp:630-638` |

P3（M60）/P4（M61）機制與架構相符但未逐行驗——兩者皆「診斷先行、10 分鐘可殺」，風險已含在 kill gate。

⚠️ 死路 ledger 的「violating boundary 修復 0 可修」「boundary-repair LP 0 可修」條**不覆蓋** M57 的重錨語意（該條只量了「block 移向邊」，M57 是「邊移向 block」）；「ordering 封卷」條**不覆蓋** M60 的 anchored 內部排序。執行 session 勿因 ledger 自我否決。

### 戰略定位（三類 ship 路徑不同，勿混）
- **M57、M59 = offline-anchor-only**（M54 in-window RED 不變）：價值 = 錨從 1.2978 再往下 + 存入「官方 RF=1.0 / median≥11s」復活條件的銀行。
- **M60、M61 = 潛在 live profile（M51 型 quality shot）**：若成立要走完整 profile 增補流程（>0.05% bar + regen 鏈），但 probe 階段先不動 shipped。
- **M58 = 唯一的 live C++ 行為修正候選**：GREEN 也只出報告，ship 與否由 A6 彙整時衡量重驗鏈成本。

---

## 1. 全域規則（每個 session 必守）

1. **禁改 shipped 檔**：`constructive.cpp` / `constructive.exe` / `optimizer_constructive.py` / `optimizer_claude.*`。要改就複製成 `constructive_m5X.cpp` → 編成 `constructive_m5X.exe`（**PowerShell 編譯**，Bash 工具寫 .exe 會失敗）。
2. **禁改既有 probe**（`m53_l3_probe.py`、`m49_refine_probe.py`、`constructive_m46.cpp` 等）：要用就複製成自己的 `m5X_*.py` / `constructive_m5X.cpp` 再改。
3. cache 各自獨立（`m5X_cache.pkl`）；`m53_l3_cache.pkl` / `m53_l2_cache.pkl` **只准唯讀**（查 winner/positions 可以，不可寫回）。
4. 產出 = repo 根 `M5X_REPORT.md`（結論 / 數據表 / kill 判定 / 新增檔案清單）＋各自 results json。**禁改 `CLAUDE.md`、`MEMORY.md`、memory/ 資料夾**（A6 統一寫）。**禁 git commit**（使用者統一處理）。
5. 走 wrapper pool / portfolio 批次時一律 `ICCAD_PROFILE_TIMEOUT=600`（本機可能同時有其他 probe session 在搶 CPU，預設 120s 會靜默丟 profile 汙染結果）。所有 probe 都是 **quality 量測**——wall-clock 不影響結論，但**勿做任何 runtime 量測**。
6. 評分一律 strict（傳 `target_positions`；`_cost_strict` 教義，勿用 `tree_decode_probe._cost_of`）。offline keep-guard 允許用官方 `evaluate_solution`（同 M53 L3 offline 錨語意）；這些 probe 永不 ship。
7. A/B 基準：offline 錨逐案值以 `results_L3_port_top32_area.json`（或 `m53_l3_cache.pkl` 內 per-case winner + LP 結果）為準；weighted bar：<0.05% = RED。
8. 幾何輸出保持 `%.17g` 精度（M10 精度牆）。

---

## 2. M57 — P1：preplaced/frozen violator 的 bbox 重錨 LP（最高優先）

**目標**：官方 boundary 相對「解自身重算的 bbox edge」（`iccad2026_evaluate.py:519-541`）。frozen violator 不動，改加等式把 bbox 邊釘到它的邊，讓其餘 movable extremes 退讓。

**做法**：複製 `m53_l3_probe.py` → `m57_reanchor_probe.py`。改 `build_and_solve` 的 `force_bnd` frozen 分支（`:278-281`）：`ui is None` 時**不** `return None`，改對其 boundary code 加 bbox 等式——
- code&1: `XMIN = x`；code&2: `XMAX = x+w`；code&4: `YMAX = y+h`；code&8: `YMIN = y`。

**實作細節（已於 2026-07-16 分析確認）**：
- (a) 某側有 frozen pin 時**跳過該側的 extreme-definer anchor 等式**（`:268-273`）——frozen pin 本身即錨（envelope 保證無人越界、frozen block 恰在邊上）；保留 mdef anchor 會過度約束（強迫舊 definer 恰停在新邊）。
- (b) 該側若有 frozen 的**已滿足** boundary block（tied 中 `u is None` → pin `bv=ext0`）→ 與重錨等式衝突 → LP infeasible = 正確幾何死，**不需特判**。
- (c) bounds（`:324-333`）已限制 bbox 變數不越過任何 frozen block → 更外側 frozen block 自動 infeasible。
- (d) 重錨到內部 frozen block 恆為 bbox **縮小**方向，與 shrink-only（`:303-305`）相容。

**Driver**：沿用 repair 模式骨架（先讀原 driver `:680-720` 的 greedy 累加邏輯），改為掃**全部 frozen violators**（202 個 violating 裡的 45 preplaced + frozen-component cluster violators），逐 block 一次 LP；keep-guard = 官方 strict cost 嚴格改善且 feasible。優先 89/85/62，再全掃 100 案。單 block 成功的案再試同案多 violator 累加。

**Kill gate**：無任何 feasible vrel 下降，或 weighted 總增益 <0.05% → RED。
**GREEN 時**：per-case 表 + 疊上現行 offline 錨（1.2978）的新合成值。
**預算**：LP-only（scipy），計算數分鐘；session 全程 ~1-2h。
**量級參考**：消一個 violation ≈ 該案 cost ×exp(−2/Nsoft) ≈ −3~5%。

---

## 3. M58 — P5：compute_nsoft 官方分母修正 probe

**目標**：量測「C++ compaction csc 改用官方 group_size−1 分母」是否改變候選選擇與分數。

**做法**：複製 `constructive.cpp` → `constructive_m58.cpp`，加 env `ICCAD_NSOFT_OFFICIAL=1`：`compute_nsoft` 的 MIB 項由 distinct-shapes−1 改為 group_size−1（**cluster 項已與官方一致，勿動**）。`=0` 時逐位=shipped（先用 1-2 案 byte-gate 驗證副本等價）。

**步驟**：
1. grep `compute_nsoft` / nsoft 全部使用點（預期僅 `compact_layout`；`ICCAD_FRAME_CSC` gated off；METRICS 輸出 cosmetic——逐一確認）。
2. 從 cons 找出有 MIB group 的案。
3. 每個 MIB 案跑其現行 winner profile（查 `m53_l3_cache.pkl` / `profile_audit.py`）env 兩檔，positions byte-diff。
4. 全同 → RED（csc 排序對分母不敏感）；有異 → 官方 strict eval 兩側，算 movers 的 weighted delta。

**Kill gate**：無 position 變化，或 weighted <0.05%。
**⚠️ GREEN 也只出報告**：ship 要付 M51 級重驗鏈（re-audit + rf_score_model regen + m49 三 gate + 官方 eval），是否划算由 A6 判。
**預算**：~30-60 min。

---

## 4. M59 — P2：REFINE rejected states → L3 種子（pilot）

**目標**：refine 丟掉的 `c2` 拓撲州收為 L3 LP 種子（機制同 L2-seeds 已證的「pre-LP 差、post-LP 贏」+0.19%）。

**做法**：複製 `constructive_m46.cpp` → `constructive_m59.cpp`，加 env `ICCAD_REFINE_DUMP=file`：逐 frame 逐 pass dump `c2` positions（%.17g）與 pre-refine `c1`。

**Driver** `m59_refine_seed_probe.py`：
1. 跑 {62,65,85,88,89,91,97} 各自 winner profile host（winner 從 `m53_l3_cache.pkl` 查）＋dump。
2. 以 L3 pair-relation signature 去重（per-pair argmax 分離軸，同 `m53_l3_probe.py:221-225` 規則），每案 cap ~8 州。
3. 每州直接過 L3 LP（把 LP 代碼複製成 m59 模組，勿 import 原檔後 monkey-patch；`--area`、2 passes）→ 官方 strict eval → 對照該案現行 offline 錨值。

**註**：`c2` 是 pre-compaction/pre-push 中間態，pilot 直接 LP、不回灌 compact/push；訊號貼近 bar 才考慮在副本 exe 加 postproc-stdin 模式回灌。

**Kill gate**：pilot weighted <0.05% 且無顯著單案 → RED（第二階段 dual-guided relation flip 一併不做）。
**GREEN 時**：報告 + 第二階段規格（|dual| top 8-12 active separation rows × 其餘 3 relations、beam 4）留待彙整後開。
**預算**：~1-2h。

---

## 5. M60 — P3：anchored first-pass 牆面容量診斷（診斷先行）

**目標**：驗證「anchored greedy（`constructive.cpp:630-688`）是否曾把後續 boundary 成員唯一可用牆段吃掉」——`boundary_penalty_est` 只看當前 block、成員序是方向不敏感的 boundary-bit + area。

**做法**：複製 `constructive.cpp` → `constructive_m60.cpp`，加 env `ICCAD_ANCHOR_TRACE=1`：anchored first-pass 每個 member commit 時 dump 其候選集（score/x/y/w/h/bp）與最終選擇。

**Driver** `m60_anchored_deficit.py`：跑硬案 winner hosts；對每個最終落在 anchored cluster 的 boundary violator，回推放置當下四邊「殘餘需求長度 − 可用牆段 interval union」deficit；檢查是否存在 selected-deficit 更差的分叉（= 當時有更保牆的替代候選）。

**Kill gate（10 分鐘版）**：不存在此類分叉 → 立即 RED，**不實作 beam**。存在 → 報告分叉清單；beam（B=4 lexicographic：第一層 `current miss + future deficit`、第二層原 score）為第二階段，probe session 不做。
**預算**：30-45 min。

---

## 6. M61 — P4：blocked exact-edge candidate → event frames（診斷先行）

**目標**：`frame_candidates()` 對 obstacle 盲（只看總面積/最大 block/preplaced extents）；量測「剛好越過 blocker 的 event frame」是否開出新可行拓撲。

**做法**：複製 `constructive.cpp` → `constructive_m61.cpp`：
1. 候選生成中 RIGHT/TOP exact-origin 候選被 overlap 擋下時，記錄越過該 blocker 的最小 ΔW/ΔH（env `ICCAD_FRAME_EVENT_TRACE=1` dump）。
2. 加 env `ICCAD_FORCE_FRAME=WxH` 強制只跑指定 frame（不讓內部 layout_score 混選——per-frame mis-rank 是已知結構，見 `constructive.cpp:1575-1586` 註解）。

**Driver**：對 {62,65,85,88,89,99} winner hosts：trace 得 event frames（每 base frame ≤2 個、去重 vs 既有 frames）→ 逐 event frame 以 FORCE_FRAME 單獨跑 → 官方 strict eval → offline 仲裁（wrapper proxy 語意）。

**Kill gate**：event 與既有 frame 重複 / 無新 feasible topology / 最佳單案 weighted <0.05%。
**預算**：30-45 min。

---

## 7. A6 — 彙整 session（**最後、單獨**跑）

前置：M57–M61 全部有 `M5X_REPORT.md`。

1. 讀五份 report，把結果寫進 `CLAUDE.md`（現況 / 未來方向 / 死路 ledger 新增對應條目；GREEN 者寫復現方式與檔案清單）。
2. **ledger 範圍更正（無論 probe 結果都要做）**：
   - 「violating boundary 修復…0 可修」條改窄：「movable violator、fixed-disjunct、non-growing bbox、逐一且須立即 official-cost 改善：0 accepted」＋補 M57 結果。
   - M26 oracle-perm 條加註：只測 yx/xy 兩 scalar key、不覆蓋 anchored first-pass 內部順序（M60 結果補上）。
   - M27 `dbg_seqpair` 條加誠實範圍註記（cluster shatter / est_cost 非官方 eval / V_rel 固定 / 預設只跑 5 案）。
3. 寫 memory 檔（每個 M 一份）＋ `MEMORY.md` index。
4. 向使用者列 commit 建議（不自行 commit）。
