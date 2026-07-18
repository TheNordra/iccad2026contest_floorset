# M63：boundary-regret beam 前置上界稽核（2026-07-19）

**判定：形式上 ABOVE BAR（T2 上界 5.11% ≫ 0.3%）→ 依預註冊規則不判 RED、本 session 不開工。但 pool oracle 反證（§7）顯示：bits 幾乎全部 pack-time 可達、卻恆付超過 violation 增益的 quality 稅——全 pool 加權可實現增益僅 0.0012%。建議 RED-leaning，開工前先過 §8 的 LP kill-test。**

## 1. 假說與判定規則

外部 codex A3：constrained/anchored placement 階段做 beam（future edge-capacity lexicographic key）可降 V_rel。依 M57-M61 教訓（行號級事實為真 ≠ 有分數），先算 weighted 上界再決定要不要開工程：

- **strict（T2）上界 < 0.3% → 免實作直接 RED**
- **≥ 0.3% → 報 per-case 目標名單與拆帳，後續 session 決定**

基準 = `results_shipped_m51.json`（beam 是送件形提案，故用 shipped 1.326473104916827 而非 L1 錨）。

## 2. 方法與 Gate 0（全綠）

`m63_vio_bound.py`：逐案從 shipped positions 以**官方 evaluator 逐位語意**重算 soft-violation numerator 並分類每個 bit（vBd 鏡射 `iccad2026_evaluate.py:519-541`：bitmask、eps=1e-6、bbox 從 solution 重算；vCl 鏡射 `:501-506` shapely unary_union；vMb 鏡射 `:511-517` round-4 distinct dims；N_soft 鏡射 `:459-471`）。

Gate 0 三項全過：
- 逐案 `evaluate_solution` 重算 cost vs json cost：**max diff = 0.000e+00（逐位）**
- 逐案 vrel / n_soft 吻合；分解三項總和 == evaluator aggregates（內建 assert，100/100）
- 加權總分重算 = **1.326473104916827**（逐位）

## 3. 分類總表（shipped M51，100 案）

| 類別 | bits |
|---|---|
| vBd 總計 | **186** |
| — frozen（preplaced violator，M57 封卷域） | 43 |
| — cluster-member（非 preplaced；其中 mixed cluster 僅 1） | **113** |
| — movable-single（其中 fixed-shape 2） | **30** |
| vCl（grouping fragments；其中 mixed cluster 16） | 19 |
| vMb | 0 |
| **V numerator 總計** | **205** |

對照 ledger 舊量測 `dbg_vio_stats`（202 = 123 cluster + 45 preplaced + 34 single）：量級一致，差異來自量測時點的 pool 演進（舊 json 為 pre-M51 某里程碑輸出）。

## 4. 三層上界（HPWL/Area/RF 不變、只削 numerator、RF=1）

`cost' = cost · exp(2·(V'−V)/n_soft)`，加權 e^(n/12)：

| Tier | 定義 | total | delta |
|---|---|---|---|
| T1 strict-single | 只歸零 30 個 vBd movable-single | 1.3079577241 | **−1.3958%** |
| T2 strict（判定用） | 歸零全部 143 個非 preplaced vBd（single + cluster-member） | 1.2587513674 | **−5.1054%** |
| T3 loose | T2 + 全部 vCl + vMb | 1.2507605890 | −5.7078% |

**T2 = 5.11% ≫ bar 0.3% → 不能免實作判死。** 注意：此為 reachability-blind 天花板——假設所有非 frozen bits 消失而 HPWL/Area 完全不動；不主張聯合可行性（同 cluster 多成員的 edge-touch 需求可能幾何互斥）。

## 5. Per-case 目標名單（T2 貢獻 top-15，dW% = 加權貢獻佔總分）

| case | n | cost | ns | frz/clu/sgl | dW% |
|---|---|---|---|---|---|
| 89 | 110 | 1.5232 | 53 | 2/**4**/0 | 0.559 |
| 85 | 106 | 1.5240 | 64 | 0/**4**/0 | 0.336 |
| 94 | 115 | 1.3132 | 65 | 0/1/1 | 0.312 |
| 76 | 97 | 1.5671 | 59 | 0/**7**/0 | 0.294 |
| 91 | 112 | 1.3255 | 58 | 1/2/0 | 0.273 |
| 98 | 119 | 1.2781 | 52 | 0/0/1 | 0.268 |
| 87 | 108 | 1.2803 | 43 | 0/1/1 | 0.252 |
| 99 | 120 | 1.3084 | 67 | 1/1/0 | 0.232 |
| 84 | 105 | 1.4223 | 62 | 0/3/0 | 0.227 |
| 88 | 109 | 1.3852 | 63 | 2/0/2 | 0.205 |
| 81 | 102 | 1.4070 | 58 | 0/3/0 | 0.186 |
| 96 | 117 | 1.3005 | 65 | 0/1/0 | 0.185 |
| 73 | 94 | 1.5715 | 56 | 0/3/2 | 0.178 |
| 80 | 101 | 1.4025 | 63 | 0/2/1 | 0.158 |
| 95 | 116 | 1.1946 | 66 | 0/0/1 | 0.154 |

共 71 案有 movable vBd bits；top-10 佔 T2 的 ~58%。主體是 **cluster-member bits（113/143）**，非 singles。

## 6. M60 交叉驗證：4/5 PASS + case 89 的 ledger 語句範圍更正

M60 名單五案（89/97/61/79/66，ledger：「連 movable violator 都沒有、全 preplaced」）：

- 97/61/79/66：**PASS**（movable=0，violators 全 frozen 或無）
- **89：MISMATCH**——有 4 個 cluster-member violator（blk13/25/63/68，全在 cluster g1）+ 2 frozen（blk23/62 = M57 記錄的那對）

**非 pool 差異**：對 `results_L1_final.json`（M60 的量測 host）重新分解，case 89 的 violator 集合與 shipped **逐位相同**（同 cost 1.523183）。解釋：g1 是**純 movable cluster**（無 preplaced 成員）→ 走複合 item packing 路徑、不經 anchored first-pass 
→ 在 M60 的 trace 域之外。M60 的診斷本身（anchored first-pass 前件空集）不受影響；但 ledger「硬案 89 …連 movable violator 都沒有」該句 shorthand 不精確——正確語意是「無 *anchored-域* movable violator」。89 的這 4 bits 正是 T2 最大單案貢獻（0.559%）。

## 7. Pool oracle：pack-time reachability 的既有證據（`m63_vio_bound.py pool`）

上界 blind 的最大問題是可達性。`audit_cache.pkl` 存有 42 profiles × 100 案的完整 positions（M51 re-audit；41 live + OM16 standby）——即現有全部 profile 多樣性（boundary-aspect、free-aspect 六子軸、frame aspects、pack-order 變體）在每案實際到達過的 layouts。逐 combo（4200）分解 movable vBd bits，與 shipped 比：

- **31/100 案存在 movable bits 更少的 profile；pool min=0 者 44/100**（29 案 shipped 本來就 0 + 15 案 pool 能全清）→ **聯合可行性不是 blocker，bits 大面積 pack-time 可達**。
- **但 29/31 案的最佳 bit-clearing profile 淨 official cost 變差**：中位稅 ~+5%、極端 case 76 +30.2%（7→5 bits）、case 65 +41.6%（7→6 bits）。
- **Case 89（T2 最大貢獻 0.559%）**：profile **#22**（`FREE_CLUSTER` wide-ratios）**全清 4 個 cluster bits**，violation 增益 exp(−8/53) = −14.0%，但 quality 稅 ~+19% → 淨 **+2.68%**。
- 唯二淨贏：case 4（#36，−2.62%）、case 28（#35，−4.72%）——**兩者皆 ORDER_SWAP 家族 = M41 從 live pool 剪掉的已知 RF trade**（quality-best `POOL=0` 1.3248 已涵蓋），不是 beam 空間的新東西。
- **全 pool 加權可實現增益 = 0.0012%**（bar 的 1/250；且全來自上述 M41 已知項）。

Caveats：n>60 的 cache rows 是無 `_band_env` overlay 的 K=12 counterfactual（非 live K=8/K=4 layout）；此偏差方向是**高估** pool 品質，不影響「bit-clearing 恆付稅」的結論方向。

## 8. 結論與建議

1. **形式判定：ABOVE BAR**（T2 = 5.11%）——依預註冊規則不免實作判死、不進 ledger、本 session 不開工。
2. **實質證據 RED-leaning**：上界是 reachability-blind 天花板；pool oracle 顯示 open question 已不是「bits 可不可達」（可達，case 89 可全清）而是「**有沒有任何 layout 清 bits 而不付超過 exp(2Δ/ns) 的 quality 稅**」——在全部 42 隻現有多樣性內答案是無（realizable 0.0012%）。機制與 M61 同構（解鎖貼牆的收益被 outline/HPWL 稅吃掉）；M53 L2（逐決策擾動全域毒）與 M27（greedy 在 quality frontier）進一步壓縮 beam 的生存空間。beam 的殘餘希望 = V_rel-directed 逐決策搜索找到 42 隻全域 env 變體沒踩到的「低稅清 bits」layout——無既有 RED 直接覆蓋，但也零正面證據。
3. **開工前的便宜 kill-test（建議後續 session 先做）**：對 pool oracle 找到的 bit-clearing 候選 layouts（如 case 89 #22、case 88/90/93/99 的 min-mv rows，positions 全在 `audit_cache.pkl`）跑 **M53 L3 LP**（凍結拓撲、exact HPWL 最小化）回收其 quality 稅：若 LP 後仍輸 shipped（稅是拓撲結構性的），beam 在該拓撲類的上限即負 → 補 RED 進 ledger；若 LP 翻正 → 才值得評估 beam 工程。此 probe 純離線、復用現成 LP 管線，不需要寫任何 beam code。

## 復現

```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m63_vio_bound.py        # 主稽核（~3 分）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m63_vio_bound.py pool   # pool oracle（~4 分）
```

產物：`results_M63_vio_bound.json`（逐案分解 + 逐 bit 記錄 + 三層上界）。
