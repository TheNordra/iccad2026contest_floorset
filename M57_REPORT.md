# M57 REPORT — frozen violator bbox 重錨 LP（P1）

**判定：RED**（kill gate 雙條件皆中：零 feasible vrel 下降、weighted 增益 +0.0000% < 0.05% bar）
**日期**：2026-07-16 · **Spec**：`M57_PLAN.md` §2 · **A/B 錨**：`results_L3_port_top32_area.json`（1.3003478581）

## 一句話結論

「邊移向 block」語意（frozen violator 不動、bbox 邊釘到它、其餘 movable extremes 退讓）**全滅**：全 100 案共 41 個 frozen violators（31 案），41/41 LP infeasible at it0、零 feasible trial、零 vrel 下降。死因經雙層鑑識分類完畢，**28/46 violated bits 是 frozen-vs-frozen 幾何死（任何語意都救不了）、18/46 是固定拓撲 separation 鏈死（單 disjunct 表達不了 relation flip，L3 已知結構限制）**。與 M53 repair（「block 移向邊」0 可修）合併後，violating boundary 修復軸在 LP 全域同動語意下**雙向封卷**。

## 實作（全依 spec，含細節 (a)-(d)）

`m53_l3_probe.py` → `m57_reanchor_probe.py`（原檔未動）：

1. **(A) frozen 分支**：`build_and_solve` 的 `force_bnd` 迴圈，`ui is None` 不再 `return None`，改依 code bits 加 bbox 等式（`XMIN=x` / `XMAX=x+w` / `YMAX=y+h` / `YMIN=y`）。
2. **(B) 細節 (a)**：sides 迴圈前算 `pin_bits`（frozen force 目標的 code 聯集），被 pin 的側**跳過 extreme-definer anchor 等式**（frozen pin 本身即錨）；tied 等式保留（mobile 已滿足 boundary block 跟新邊同動；frozen tied 的 `BV=ext0` 與 pin 衝突 → infeasible = 細節 (b) 正確幾何死）。
3. 細節 (c)(d) 零程式碼：bounds 已擋更外側 frozen block；重錨恆縮 bbox、與 shrink-only 相容。
4. **driver `reanchor`**：repair 骨架、逐案 loader gate（eval==json cost assert）、frozen 判定鏡射 `build_and_solve`（preplaced ∪ 含 preplaced 的 cluster component）、greedy 累加、keep-guard = 官方 strict `evaluate_solution`（feasible ∧ cost 嚴格改善）、案序 89/85/62 優先。
5. **鑑識**：`diag`（靜態死因分類）、`diag2`（LP 級：只 pin violated bits／逐 bit 單獨 pin；`force_bnd` 擴充支援 `(block, mask)`）。

**驗證鏈**：gate0 100/100 exact（loader/eval 未破）→ 全掃 → refactor 後 case 89 重跑逐位同 → diag/diag2。

## 數據

### 全掃總表（`reanchor --area`，src = top32_area 錨）

- frozen violators tried **41**（31 案；45 preplaced violators 中 41 個在錨 layout 仍 violating，含 frozen-component cluster 成員）
- kept **0** / feasible-but-rejected **0** / feasible vrel-drop trials **0** / LP dead：`{'lp_status_2': 41}`（HiGHS infeasible）
- weighted gain **+0.0000%** → 新總分 = 錨 1.3003478581 不變；offline 合成錨維持 **1.2978**（RED 故未跑 l2stack 疊算）

31 案逐案（cost/vrel 全數不變，僅列 violator 明細）：

| case | n | viol | frozen tried | LP 結果 |
|---|---|---|---|---|
| 89 | 110 | 2 | blk23(c6), blk62(c2) | 全 INFEAS |
| 86 | 107 | 2 | blk0(c4), blk100(c4) | 全 INFEAS |
| 88 | 109 | 4 | blk31(c6), blk73(c2) | 全 INFEAS |
| 90/91/92/93/99 | 111-120 | 1-3 | 各 1-2 隻 | 全 INFEAS |
| 3/7/12/16/17/23/30/31 | 24-52 | 1-2 | 各 1 隻 | 全 INFEAS |
| 49/52/53/54/57/59/61/63/64 | 70-85 | 1-5 | 各 1-2 隻 | 全 INFEAS |
| 68/69/70/74/77/78 | 89-99 | 1-4 | 各 1-2 隻 | 全 INFEAS |

（85 與 62 在錨 layout **無 frozen violator**——其 vBd 殘留全是 movable/cluster violators，屬 M53 repair 已判死範圍。）

### 死因分類（diag 靜態 + diag2 LP 級交叉驗證）

46 個 violated bits（41 violators；另 11 個 touch bits 的 pin 為 no-op 非死因）：

| 死因 | bits | 說明 |
|---|---|---|
| **(c) 更外側 frozen block** | 28 | 另一 preplaced/frozen 在該側伸得更外 → recomputed edge 永遠到不了 violator（bounds 衝突）。其中 27 個同時有 **(b) 同側 frozen 已滿足 boundary block** 釘死舊邊。**任何語意都救不了**（frozen-vs-frozen）。 |
| **chain（固定拓撲鏈死）** | 18 | 無 (b)(c)，死於 separation 單 disjunct：擋路的 movable 群要讓邊進來需 relation flip，LP 拓撲凍結表達不了（同 plan §0 已驗事實「每 pair 只留單一 max-gap disjunct」）。dist 0.095~62.3 皆死——**連 0.095 單位的重錨都不可行**（case 99 blk24 XMIN，見下）。 |

diag2 決定性排除 artifact：

- **只 pin violated bits（拿掉 touch pins）**：41/41 仍 INFEAS → touch-bit pin 不是死因。
- **逐 bit 單獨 pin**（多 bit violators）：全 INFEAS，唯一例外 **case 99 blk24 XMIN 單獨 pin FEAS**（dist 0.095）——但該 violator 的另一 bit YMAX 鏈死，單 bit 滿足不清 violation → vrel 0.0448 不變、cost 1.299369→1.299388 反升 → keep-guard 正確拒絕。此例同時證明 pin 等式建構本身正確（可行時 LP 解通過官方 strict eval feasible）。

### 量級對照

Spec 預估消 1 個 violation ≈ 該案 −3~5%；實際可實現數 = **0**。headroom 全部被 frozen 幾何（61%）與拓撲凍結（39%）鎖死。

## Kill 判定

- gate 條件 1：無任何 feasible vrel 下降 → **中**（0 trials）。
- gate 條件 2：weighted <0.05% → **中**（+0.0000%）。
- → **RED**。不進 GREEN 分支（無 per-case 改善表、無合成錨更新、不跑 l2stack）。

## 給 A6 的 ledger 更正素材

「violating boundary 修復 0 可修」條現在可寫成**三語意封卷**：

1. 單 block 移向邊（`dbg_vio_stats.py`，202 個 0 可修）；
2. LP 全域同動、block 移向邊（M53 L3 repair，0 可修）；
3. **LP 全域同動、邊移向 frozen block（M57，本報告）：41/41 infeasible、0 可修**。死因：28/46 bits frozen-vs-frozen 幾何（(b)+(c)）＝語意無關的絕對死；18/46 bits 固定拓撲鏈死＝**只在 L3 拓撲凍結語意下死**，relation-flip 類方法（M59 第二階段 dual-guided flip）理論上仍可攻，但 movable violators 已由 M53 證死、frozen violators 的 61% 是絕對死 → 期望值極低。
4. 附帶：案 85/62 的 vBd 殘留無 frozen violator（全 movable/cluster），案 89 的 2 個 frozen violators 一個絕對死（blk62，(b)+(c)）一個鏈死（blk23，dist 62.3 = 結構性）→ 三大殘留案此軸徹底無肉。

## 新增檔案

| 檔案 | 說明 |
|---|---|
| `m57_reanchor_probe.py` | m53_l3_probe.py 副本＋M57 修改；新 modes：`reanchor`（主量測）、`diag`（靜態死因）、`diag2`（LP 級逐 bit 鑑識）；`force_bnd` 支援 `(block, mask)` |
| `results_M57_reanchor.json` | 全掃 dump（0 movers → 與 src 錨逐案同值，留作 schema 證據） |
| `M57_REPORT.md` | 本報告 |

無 cache 檔（LP 全流程 <10s，不需斷點續跑）。未動任何 shipped 檔 / `CLAUDE.md` / `MEMORY.md` / `memory/`；未 commit。

## 復現

```powershell
$py = "C:\Users\Nordra\.conda\envs\iccadv\python.exe"
& $py m57_reanchor_probe.py gate0 --anchor results_L3_port_top32_area.json  # 100/100
& $py m57_reanchor_probe.py reanchor --area                                 # 主量測（RED）
& $py m57_reanchor_probe.py diag                                            # 靜態死因
& $py m57_reanchor_probe.py diag2 --area                                    # LP 級鑑識
```
