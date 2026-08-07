# M78 — 候選集合的第二條路徑（candidate-set second path）

> 2026-08-03。**判定 RED**（`anch_cross` OOS −0.160% vs bar 0.30%）。
> 工具 `m78_antecedent_census.py`、`m78_probe.py`、`m78_oracle.py`、`m78_diff.py`、
> `constructive_m78.cpp` / `.exe`（全部離線，永不送件）。
> **送件形零改動**：出貨的 `constructive.exe` md5 未變（`a576feb6…`），
> `optimizer_constructive.py` 未動，三顆離線 cache 全部仍然有效。

## TL;DR

| 結論 | |
|---|---|
| **11 個候選機制，10 個變差、1 個變好** | 唯一贏的是 `anch_cross`（mixed cluster 的跨 rect 交叉候選）：in-set −0.183%、OOS 240@48c **−0.160%**、wall +7% ⇒ **RED**（bar 0.30%） |
| **M71 的排序增益搬不過來** | anchored 路徑沒有內部版型可排列 ⇒ `ord1` 恰 0、`ord2/3` 只動 3 案。M71 的本體是**內部版型枚舉**不是排序 |
| **「加候選」預設有害** | 同一個交叉機制：anchored −0.18%、泛用 **+0.36%**；中心對齊槽兩邊都是正的 ⇒ 出貨的候選集合是**調過的**不是貧乏的 |
| 🚨 **抓到一個會製造假 RED 的 gate 缺陷** | `_m75_liveness()` 餵 `target_positions=None` ⇒ 刪掉所有 preplaced ⇒ **anchored 路徑的旗標前件恆為空**。連帶使 M75 對 `ANCHORED_BND_REPACK` 的「恰 0.0000%」需要重測（見 §4b / §6） |

## 0. 為什麼是這個軸

`constructive.cpp` 的 cluster 有**兩條互斥路徑**（`solve():1616-1627`）：

| 路徑 | 條件 | 候選集合來自 | M71 |
|---|---|---|---|
| 純 movable cluster | `pre.empty()` | `make_group_item():354-566` | ✅ 改了候選集合＋排序 key → in-set −1.589% / OOS −4.04% |
| **mixed（preplaced+movable）** | `!pre.empty() && !mov.empty()` | `adjacent_candidates_for_block():692-727` + first-pass `pack_in_frame():792-856` | ❌ **沒動過** |

M71 之後留下的三個**從未被參數化的結構決策**：

1. **anchored 成員排序 key**（`pack_in_frame:796-800`）＝ `(block_boundary_score desc, area desc)`，
   **只有一個順序、沒有仲裁**。對照組：`make_group_item` 枚舉最多 **7 種**成員順序 ×
   5 種內部版型，再用 `(fragments, boundary_bad, area, aspect)` 挑 lex-best。
2. **候選位置集合**：`adjacent_candidates_for_block` 每個 rect 只給 **8 個角對齊** abutment 槽；
   泛用的 `item_candidates:648-657` 也一樣（另加 4 個 frame 角 + boundary 專用 xv×yv）。
   **沒有中心對齊槽**、**x 與 y 永遠取自同一個 rect**（跨 rect 交叉候選不可達）。
3. **greedy 評分的位置偏置** `+1e-3*y + 1e-4*x`（`:841`/`:907`/`:975`）——寫死的左下優先，從未變動。

**為什麼這不在已封卷的範圍內**：M26 封的是 **ordering**（完美排序只值 +0.005%）、
M27 封的是 **packer 典範**（B\*-tree/SP）、M61 封的是 **frame**、M33-M39 掃的全是**成員 aspect**。
**沒有任何上界涵蓋「同一個 greedy 內、更豐富的候選位置集合」**——那正是 M71 唯一動過的東西。

## 1. Step 0 — 前件普查（`m78_antecedent_census.py`，讀 dataset、不跑 placer）

ledger 的 `[[m57-codex-gap-plan]]` / `[[m60-anchored-wall-red]]` 教訓：**行號級事實為真 ≠ 有分數**，
M60 就死在前件空集。所以先數前件，`exp(n/12)` 加權，並用 **M71 的前件當尺**（唯一知道匯率的軸）。

| 語料 | A1（anchored movable）佔加權 blocks | 非空案加權占比 | M71 路徑 | **A1/M71** |
|---|---:|---:|---:|---:|
| in-set 100 | **6.26%** | 73.17%（53/100 案） | 17.59% | **0.36×** |
| OOS s1 240 | **5.66%** | 73.56%（139/240） | 18.71% | **0.30×** |
| OOS s2 240 | **5.86%** | 62.27%（129/240） | 18.17% | **0.32×** |

- 重帶 **n>100**（81-90% 的權重）：A1 非空 **14/20**（in-set）、**56/80**（s1）、**50/80**（s2）。
- 其中 **boundary!=0** 的 anchored 成員 = 1.70/1.90/1.78% of blocks（EXPOSE/EDGE_PACK 鏡像的標的）。
- 泛用 singles 路徑（A2）= **77-79% of blocks，100% 的案子**，前件是 A1 的 **12.6 倍**。

⚠️ **誠實範圍**：**A1 與 M71 幾乎不互斥**——「有 A1 前件但 M71 前件為空」的案子只佔
**0.00% / 2.55% / 3.59%** 的權重。也就是說 A1 要贏的是 M71 已經改善過的同一批案子，
**不是** 一塊沒人碰過的新地。block 層面兩者是互斥的（一個 block 不會同時屬於兩種 cluster），
但**案子層面高度重疊** ⇒ 邊際報酬可能低於 0.30-0.36× 這個 block 比例的線性外推。

**判定：Step 0 PASS**（kill bar 是加權覆蓋率 < ~5%）。

## 2. Step 1-2 — 儀裝副本與 Gate 0

`constructive_m78.cpp` = `constructive.cpp` 複本 + 六個旗標，**全部預設 0**：

| 旗標 | 軸 | 機制 |
|---|---|---|
| `ICCAD_M78_ANCH_ORD=1..4` | A1 | anchored 成員排序 key：1=corner-first、2=L/R/B/T、3=R/L/T/B、4=純 area（對照組） |
| `ICCAD_M78_ANCH_CENTER` | A1 | `adjacent_candidates_for_block` 加中心對齊 abutment 槽 |
| `ICCAD_M78_ANCH_CROSS` | A1 | 加跨 rect 交叉候選（x 取自一個 rect 的面、y 取自另一個） |
| `ICCAD_M78_ITEM_CENTER` | A2 | `item_candidates` 同樣的中心對齊槽 |
| `ICCAD_M78_ITEM_CROSS` | A2 | `item_candidates` 的跨 rect 交叉，限 frontier 最近 `ICCAD_M78_CROSS_K`（預設 6）個 rect |
| `ICCAD_M78_TIEBREAK=1..3` | A3 | 位置偏置：1=x-major、2=對角、3=右上 |

**兩個設計約束**（都不是可選的）：

- 新候選一律 **append 在 dedup 之前**，`clamp → unordered_set dedup → sort` 的尾段一字未動
  ⇒ M46 的逐位保證所依賴的比較器沒被碰過（`[[m46_wall_setter_speedup]]`）。
- 全部只進 `constructive_m78.exe`，**出貨的 `constructive.exe` md5 不變**。自 M74 起所有離線
  cache 的簽章都釘出貨 exe md5，動它就要重建 `audit_cache.pkl` / `audit_cache_ship.pkl` /
  `audit_cache_esc.pkl`（各 8-11 分、必須序列）再重跑四個 gate。**那筆錢等 GREEN 再付。**

### Gate 0（`m78_probe.py gate0`）

100 案 × **35 隻** pool profile（12/48 核池的聯集，由 `oc._pool_indices()` 產生，
**不可手拼**——M41 的 swap 過濾是依內容的）× 兩顆 binary，比 stdout 的 md5
（輸出是 `%.17g` 文字 ⇒ 位元組相等即逐位相等）：

```
GATE 0  100 cases x 35 pool profiles = 3500 pairs
  failed runs        : 0
  differing outputs  : 0
GATE 0: PASS (m78 flags-off is bit-identical)
```

⇒ 後面所有數字都站在「m78 旗標全關 ≡ 出貨 binary」這個地基上。這也讓
`m67_oos_probe.py --arm-bin` 的做法合法：**arm 側跑 m78 exe、shipped 端點仍沿用 cache 裡
constructive.exe 的解**（arm 名附掛 probe binary 的 md5，重編後不會誤用舊解）。

## 3. Step 3 — liveness（進行中）

> `[[m75-m71-residual-knobs-red]]`：**liveness 不可用 portfolio 輸出判**。旗標能改候選卻改不動
> proxy argmin ⇒ 四個旗標都會回報「零差異」的**假 RED**。要用 **per-profile binary 輸出**；
> 而且一旦證明某案全池 profile 逐位相同，該案 portfolio 就**可證明**不變 ⇒ 只解活案即得精確 delta。

100 案 × 35 隻 pool profile，旗標 on/off 比 stdout md5：

| arm | 活 (case,profile) | 活案 | 活案加權 | 最大 n |
|---|---:|---:|---:|---:|
| `anch_ord1` corner-first | **0/3500** | **0/100** | **0.00%** | — |
| `anch_ord2` L/R/B/T | 73/3500 | 3/100 | 0.92% | 91 |
| `anch_ord3` R/L/T/B | 73/3500 | 3/100 | 0.92% | 91 |
| `anch_ord4` 純 area（對照組） | 1009/3500 | 32/100 | 54.63% | 119 |
| `anch_center` | 1232/3500 | 48/100 | 65.79% | 120 |
| `anch_cross` | 810/3500 | 45/100 | 65.63% | 120 |
| `item_center` | 3422/3500 | 100/100 | 100.00% | 120 |
| `item_cross` | 3086/3500 | 100/100 | 100.00% | 120 |
| `tb1` x-major | 1018/3500 | 99/100 | 100.00% | 120 |
| `tb2` 對角 | 751/3500 | 98/100 | 99.99% | 120 |
| `tb3` 右上 | 1658/3500 | 98/100 | 99.99% | 120 |

### 🔑 結構性發現：M71 的排序增益**搬不過來**

`anch_ord1` **恰 0**（與 M75 的 CORNER/REPACK 同一個現象：`block_boundary_score` 已經把
corner block 排在最前，corner-first 與它重合），`ord2`/`ord3` 只在 **3 個案子**（27/54/70）
動、佔 **0.92%** 的權重。

原因是結構性的：**anchored 路徑沒有內部版型可以排列**。`make_group_item` 之所以能從 7 種
成員順序拿到分，是因為每種順序都會餵出 5 種不同的**內部版型**（shelf/column/square/wide/two-rows），
再用 layout key 挑；而 anchored 成員是**逐一直接放進 frame**，順序只在兩個成員的 boundary
score 打平時才有差別——那很罕見。⇒ **M71 的增益本體是「內部版型枚舉」，不是「排序」**，
這一半在 mixed cluster 上不存在。

還活著的是另外兩件事：**候選位置集合**（center / cross）與**位置偏置**（tiebreak），
兩者都不需要內部版型。

## 4. Step 4 — in-set 精確 delta（待跑）

錨 = `results_M74_default.json` / **1.293461035226291**。

**先驗證計分管線**（M77 `selftest` 的同一種紀律）：`m78_probe.py score off` = m78 exe、
旗標全關、跑真 wrapper 的 41 隻 portfolio、官方 `evaluate_solution` 計分 ⇒
**`1.293461035226291`，delta −0.0000%，100/100 feasible**，與錨逐位相同。
⇒ 後面每個 arm 的 delta 都是純粹來自旗標。
（⚠️ `solve()` 要餵 `build_opt_target_pos()` 的**遮罩版** target_positions，不是 `fp_sol`
本身——官方 harness `:866-881` 只給 preplaced `(x,y,w,h)` 與 fixed `(w,h)`，那是硬約束的
**輸入**不是答案；餵原始 tp 就是 label 洩漏。）

⚠️ **in-set 只是篩子不是判準**——M75 的 PERMUTE in-set **+0.0104%（正）**、OOS **−0.0111%（負）**，
符號會翻。

### 結果（10 個活 arm，全部 100/100 feasible）

| arm | total | delta（**正 = 更差**） | wall |
|---|---:|---:|---:|
| **`anch_cross`** | **1.291096744787358** | **−0.1828%** ✅ | 141s（off 142s，中性） |
| `anch_ord2` | 1.293589230976189 | +0.0099% | 143s |
| `anch_ord3` | 1.293589230976189 | +0.0099% | 142s |
| `anch_center` | 1.297154266085060 | +0.2855% | 143s |
| `tb1` x-major | 1.294372019545187 | +0.0704% | 142s |
| `tb2` 對角 | 1.294801956062232 | +0.1037% | 142s |
| `tb3` 右上 | 1.297997383271214 | +0.3507% | 142s |
| `item_cross` | 1.298150061156868 | +0.3625% | **186s（+31%）** |
| `item_center` | 1.298587092380962 | +0.3963% | **204s（+44%）** |
| `anch_ord4`（對照組） | 1.307288499703665 | +1.0690% | 143s |

### 🔑 兩個一般性結論

1. **「加候選」預設是有害的，不是有益的。** 完全相同的跨 rect 交叉機制，放進 anchored
   路徑是 **−0.18%**、放進泛用路徑是 **+0.36%**；中心對齊槽兩邊都是正的（+0.29 / +0.40%）。
   ⇒ 出貨的候選集合**不是貧乏而是調過的**：greedy 的 `bbox_area_with` 是短視的，多給它
   局部更好的位置反而讓它做出全域更差的選擇（M52「零容錯帶」的另一個面貌）。
   **這條否證了「候選集合越大越好」的樸素版假說**，只有一個特定補洞有用。
2. **`anch_ord4` +1.0690%** 是有用的對照組：把 boundary-first 優先序拿掉就掉 1%
   ⇒ 那個排序是**承重的**，不是隨手寫的。配合 `ord1` 恰 0 / `ord2,3` 只動 3 案，
   anchored 排序軸整條**封卷**。

### `anch_cross` 逐案（19 movers：7 好 / 12 壞）

| case | n | off | anch_cross | d% | 加權貢獻 |
|---:|---:|---:|---:|---:|---:|
| 90 | 111 | 1.33009 | 1.29443 | **−2.681%** | −0.1042% |
| 93 | 114 | 1.26123 | 1.23713 | −1.911% | −0.0904% |
| 86 | 107 | 1.33965 | 1.29893 | **−3.039%** | −0.0852% |
| 83 | 104 | 1.29496 | 1.27954 | −1.191% | −0.0251% |
| 95 | 116 | 1.19465 | 1.19126 | −0.284% | −0.0150% |
| …（另 2 個微幅變好） | | | | | −0.0005% |
| 76 | 97 | 1.30058 | 1.31775 | +1.321% | +0.0156% |
| **88** | 109 | 1.38518 | 1.42590 | **+2.939%** | **+0.1007%** |

**機制吻合度高**：90 / 86 / 88 / 95 正是 Step 0 普查裡 anchored 質量最大的那幾案
（anchMov 23 / 15 / 15 / 15）⇒ 這不是噪聲，是機制真的在它該作用的地方作用。

⚠️ **但淨值是靠 case 90 與 case 88 互抵後的殘量撐著**，符號脆弱。而且全域 overlay 形式下
**被弄壞的 12 案沒有逃生口**（41 隻 profile 每一隻都帶機制）。
**2-way per-case oracle 上界 = −0.3204%**（只取 7 個變好的），而 M76/M77 已證
**proxy 在異質候選上是 oracle-perfect** ⇒ 若改成 pool tier 形式理論上可realize 到 −0.32%，
代價是 wall。**先看 OOS 再決定要不要走那條**。

## 4b. 🚨 途中抓到的 gate 缺陷：`_m75_liveness()` 把 preplaced 整個丟掉

跑 `m67_oos_probe.py restore --arm m78anchcross` 時 GATE A 回報
**`0/210 (case,profile) pairs move`**，與我方 `m78_probe.py live` 量到的「45 案活」
直接矛盾。根因（`m67_oos_probe.py:1043`，M75 寫的）：

```python
inp = _serialize_input(n, ..., lay["cons"], None, gnn_hint=None)
#                                            ^^^^ target_positions
```

`target_positions=None` ⇒ 序列化出來的輸入裡**沒有任何 preplaced `(x,y,w,h)` 與 fixed `(w,h)`**。
沒有 preplaced 就**沒有 mixed cluster**（`solve():1619` 的 `!pre.empty()` 恆偽）⇒
**任何作用在 anchored first-pass 的旗標，前件恆為空**，gate 會回報一個看起來完全合理的 0。

診斷（同一支 binary、同一組 profile，只換輸入）：

| case | n | `otp`（正確） | `None`（gate 原本餵的） |
|---:|---:|---:|---:|
| 54 | 75 | **24/35 moved** | 0/35 |
| 90 | 111 | **34/35 moved** | 0/35 |
| 86 | 107 | **19/35 moved** | 0/35 |

⇒ 這是「**量測組態 ≠ 部署組態**」的同一族 bug（組員交接文件 §2、我方 M74 的
`_m71_env()` 漏帶）。已修：改餵 `build_opt_target_pos()` 的遮罩版，與 `_solve_one()`
餵真 harness 的完全一致。

另外兩個同一輪修掉的坑：

- **相對路徑的 `--arm-bin` 會靜默失敗**：Windows 的 `CreateProcess` 先搜 `python.exe` 的
  目錄再搜 cwd ⇒ 裸檔名啟動失敗、`_run_profile` 吞掉例外回 `None`、兩側都是 `None`
  ⇒ 又是一個好看的 0。已改 `Path(...).resolve()` **並在 liveness 迴圈加 assert**
  （launch 失敗必須炸，不可與「旗標無效」混為一談）。
- **壞結果被寫進 liveness cache 後被下一輪重用**：`lk` 只釘 arm 名 + exe md5，不釘
  「這個 screen 餵什麼給 binary」。已升到 `@v3` 並在註解寫明**每次改 screen 的輸入
  就要 bump**。

### M75 的 REPACK 重測：**判定活下來，但數字要更正**

M75 判 `ICCAD_ANCHORED_BND_REPACK` 為「**恰 0.0000%，340 案零 profile 輸出改變**」，
而 REPACK 正是作用在 anchored first-pass 的旗標（`pack_in_frame:844-847`）——
正好是被上面那個缺陷清空前件的那條路徑。所以它必須用正確輸入重量。

用 `m78_probe.py live repack corner`（100 案 × 35 profile、`build_opt_target_pos` 的遮罩 otp）：

| 旗標 | M75 記載 | **M78 重測（正確輸入，in-set 100 × 35 profile）** |
|---|---:|---:|
| `ANCHORED_BND_REPACK` | 0（340 案） | **1/3500 (case,profile)**、1 案（case 75）、1.08% 權重 |
| `CLUSTER_BND_CORNER`（對照組） | 0（340 案） | **0/3500** — 原判定原封不動 ✅ |

⇒ **M75 的結論成立，敘述要改**：不是「恰 0」而是「3500 對裡動 1 對」，實務上仍 inert，
而且 M75 給的**機制解釋正是這個數字的原因**——REPACK 的 ±9000 偏置對既有的
`BP_W*bp`（=30000）幾乎恆為保序變換，argmin 幾乎不可能翻。所以那條 RED **不需要翻案**，
但「恰 0.0000%」這個措辭是缺陷造成的，應改為「≈0（1/3500）」。

CORNER 本來就不受此影響：它在 `make_group_item`，而 `None` 輸入會讓**更多** cluster
變成純 movable ⇒ 前件只會變大不會變小。

**教訓**：一個 gate 缺陷可以讓一個**正確的**結論建立在**錯誤的**證據上。
兩者要分開處理——修缺陷、重量、然後才知道結論要不要動。

## 5. Step 5-6 — OOS 240 @48 核 + wall（`anch_cross`，唯一走到這關的 arm）

`m67_oos_probe.py restore --arm m78anchcross --arm-bin constructive_m78.exe
 --force-cores 48 --pool0-lo 0`（GATE A 七項全 PASS，liveness `102/210`）。

| 量 | 值 |
|---|---:|
| in-set @16 核（tier-3 ON） | **−0.183%** |
| in-set @48 核（tier-3 OFF、tier-5 ON） | **−0.213%**（21/100 movers） |
| **OOS 240 @48 核** | **−0.1604%**（**33 好 / 40 壞**，0 infeasible） |
| wall @12c | 2.13 → 2.28s（**1.07×**）；RF sign-check **上界 +2.013%** |
| **bar** | NET ≥ **+0.30%** |

**判定：RED。** 品質單獨就只有 bar 的 **54%**，而 wall 是扣分不是加分。

⚠️ 但這是**一個不同種類的 RED**，值得分清楚：機制本身是真的。
- **in-set → OOS 轉移率 76%**（0.1604 / 0.2130），遠高於 M76 量到的 ≈5%
  ——因為這是**機制**不是**樣本內挑出來的 source set**。
- movers 落在普查預測的位置（in-set 90/86/88/93/95 = anchored 質量最大的案子）。
- 它就是**太小**：mixed cluster 只佔 5.7-6.3% 的加權 blocks，而其中一半的案子會變差。

### 5.1 pool-tier 形式的上界（`m78_oracle.py`）

全域 overlay 的結構缺點是**被弄壞的 40 案沒有逃生口**。tier 形式（加 knob-ON twin、
讓 proxy 逐案仲裁）的上界 = 2-way per-case oracle：

```
shipped               1.555855
arm (global overlay)  1.553360   +0.1604%
2-way per-case ORACLE 1.549879   +0.3841%   <- tier 形式的天花板
realized by overlay: 41.8% of the ceiling
```

**天花板 +0.384% 仍然不值得建**，三個理由都是既有量測：

1. 那個上界要**每一隻 host 都配一個 twin**（41 隻 ⇒ 池翻倍）。48 核 wall = max-setter
   （M67-E，100/100），池有 13 個空位；twin 比 host 慢 **7%** ⇒ 只要 max-setter 的 twin
   進池，`ΔRF = 1.07^0.3` = **+2.04%**，一口氣吃掉全部品質增益。
2. 要避開那個，就得挑「不會變成 max-setter」的子集 ⇒ **靠 in-sample 貪婪選 source set**，
   而 M76 已量到那種選法的**優勢轉移率 ≈5%**（`m73x` vs `m73big` 的乾淨對照）。
3. 直接對照組：M76 的 escape tier 用 4 隻 source，同樣的 OOS×48c 條件下只 realize 到
   **+0.101~0.107%**。

⇒ 與 M76 同一個判準、同一個結論。**不建 tier。**

### 5.2 s2 為什麼不跑

s2（worker_10..19）是為 **ML 候選**建的——s1 抽自 `floorset_lite` worker_0..9 ＝ 組員
ML 的訓練語料，對他們的模型是樣本內。M78 是純古典 C++ 旗標，**s1 對我方一直是乾淨的
held-out**，也是 M75/M76 用的同一把尺。品質連 bar 的 54% 都不到，加第二份語料改變不了判定。

## 6. 結案與留下的東西

**送件形零改動。** 出貨 `constructive.exe` md5 仍是 `a576feb6…`、`optimizer_constructive.py`
未動、`audit_cache*.pkl` / `m67_oos_cache*.pkl` / `m77_oos_audit.pkl` 三顆離線 cache
全部仍然有效（**這正是把所有實驗關進 `constructive_m78.exe` 的目的**——GREEN 了才付重建
三顆 cache ＋ 重跑四個 gate 的錢）。`m67_oos_cache_c48.pkl` 只被**增量**寫入
（新的 arm key 與 `m75_live@v3` key），shipped 端點仍驗到 `1.555855`，與 M76 錨一致。

### 留在 tree 上的工具（永不送件）

| 檔 | 用途 |
|---|---|
| `m78_antecedent_census.py` | 前件普查（`inset\|s1\|s2\|all`），讀 dataset、不跑 placer。**任何新軸開工前先跑這支** |
| `constructive_m78.cpp` / `.exe` | 六旗標儀裝副本，全部預設 off、Gate 0 逐位相同 |
| `m78_probe.py` | `gate0`（3500 對逐位）/ `live`（per-profile liveness）/ `score`（真 portfolio + 官方計分；`score off` 是自檢，逐位重現錨） |
| `m78_oracle.py` | 從 A/B dump 算 2-way per-case oracle = **pool-tier 形式的天花板** |
| `m78_diff.py` | 兩份 `score` dump 的逐案 movers |
| `m67_oos_probe.py --arm-bin` | **arm 側跑另一顆 binary、shipped 端點沿用 cache**；arm 名附掛 binary md5。這是「新 C++ 軸不動出貨 exe」的標準做法 |

### 下一個人要知道的三件事

1. **候選集合這條軸關了。** 兩條路徑 × 三種加法 × 三種偏置全量過；唯一有量的
   `anch_cross` 只有 bar 的 54%，tier 形式的天花板 +0.384% 也買不起 max-setter twin 的
   ΔRF +2.04%。
2. **前件普查很便宜，一定要先跑，但它量不到全部。** Step 0 幾分鐘就給出「A1 的前件是
   M71 的 0.30-0.36×」這把尺；而最終增益比（0.160 / 4.04 = **0.04×**）比那把尺**還低
   一個數量級**——因為 M71 的增益本體（內部版型枚舉）在這條路徑上根本不存在。
   **普查量得到前件大小，量不到機制能不能搬過來**；後者要靠讀懂機制本身。
3. **修 gate 缺陷 ≠ 翻案**（見 §4b）：REPACK 的證據是錯的，結論是對的。
