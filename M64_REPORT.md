# M64 — L3 相鄰拓撲 probe（單 unit-pair separation relation flip 後 LP 重解）

**判定：RED（pilot gate，2026-07-19）。** 3 案（85/88/91）× top-24 unit pairs × 3-4 替代
relation × 2 variants = **529 個 flip probe，0 movers**（官方 cost 改善 >1e-6）→ 依預註冊
pilot gate 停止，不擴 heavy/full、不需 l2base 誠實基準（union gain 恆 0）。
offline 錨維持 **1.2978**、送件形維持 M51 shipped 不動。

## 1. 定位

M53 L3 對每個 block pair 只保留「目前 max-gap 的單一 separation disjunct」
（fixed-disjunct，`m53_l3_probe.py:213-231`）→ 1.2978 是既有種子所落拓撲 cell 的連續最優，
相鄰 cell 從未量測。M64 = M59 stage-2（relation flip）的正式量測——當時依 kill gate 跳過
並非測死；起點 = 錨自身 cell（`results_L3_port_top32_area.json` 逐案 winner positions），
非 M59 c2 遠盆地種子。

## 2. 方法（`m64_flip_probe.py`）

- **Flip 語意 = unit-pair 級**：目標 = unit pair (A,B)（rigid cluster component / free single /
  **per-block frozen pseudo-unit** `('F',blk)`）+ canonical 方向 k∈{A<B, B<A, AvB, BvA}；
  所有跨 (A,B) 的 block pair 的 separation row **替換**成方向 k（各自 gap 常數；block 順序
  映到 (B,A) 時鏡射 0↔1/2↔3）。單 block-pair flip 在多成員 unit 間近必矛盾（同 unit 其他
  成員 pair 保留原 disjunct）故不採。
- **排序**：score = (A↔B 跨邊 Σw + 兩端 incident live Σw) / (max(cur_gap,0) + 1e-3·(W0+H0)/2)，
  取 top-24；成員 argmax 方向異質的 pair 試全 4 方向（統一本身即拓撲變更），同質試 3 個
  非當前方向；與 baseline `max()`（first-wins tie-break）逐成員相同的方向剔除（exact no-op）。
- **每 (pair,dir)**：sound 前濾（`extent(A)+extent(B) > bbox row` 才跳、證明不可行不漏 mover）
  → **variant A**（strict，bbox 不增長）→ **variant B**（A infeasible 或 |delta|<0.2%·base 時，
  bbox 兩 row 放寬 ×√1.005 → area 增長 ≤0.5%；四邊全被 frozen 等式釘死則跳過）→ 官方
  strict `cost_eval` 仲裁；mover 另跑 `m53.lp_pass(area_obj=True)` fixpoint polish（≤3）。
- LP 機構 = `m53_l3_probe.build_and_solve` 逐行複製 + `force_rel`/`skip_bnd_ties`/`bbox_relax`
  三參數（原檔未動）；cluster 剛體 ladder、boundary 等式、envelope、HiGHS 全同 M53。
- **Selfcheck PASS**：錨 3 案 `cost_eval` 逐位 = json；對 homogeneous pair 強制其**當前**方向
  → 解與無 force **逐位相同**（force_rel 佈線證實）。

## 3. 數字（pilot：529 flips + 15 diags，LP wall 360s）

| case | n | 錨 cost | 候選(A+B) | LP infeasible | prefilter | feasible | 最佳 delta | feasible 稅 (med/max) |
|---|---|---|---|---|---|---|---|---|
| 85 | 106 | 1.458195 | 192 | 152 | 8 | 32 | **−8.2e-13**（tie 級） | −8e-13 / +1.19e-3 |
| 88 | 109 | 1.325391 | 155 | 148 | 0 | 7 | ±0.000000 | +4.1e-4 / +2.81e-3 |
| 91 | 112 | 1.296344 | 182 | 159 | 0 | 23 | ±0.000000 | 0 / +4.47e-3 |

- **狀態直方圖**：`lp_status_2`（infeasible）459 = **86.8%**；feasible-but-worse/tie 62 = 11.7%；
  prefilter 8 = 1.5%。**零 ladder kill、零 cluster_break**（全部 attempt-1 純幾何判定）。
- **62 個 feasible 的 vrel delta 全部恰 0**（0/62 nonzero）——相鄰拓撲 cell 完全動不了
  violation bits；cost 變化純 quality 項。
- **稅種**：mean d_hgap **+3.2e-4** ≫ d_agap +8e-5、d_vrel 0 → **HPWL 稅主導**（非 area 稅）。
- 全 pilot 最佳結果 = case 85 `(('F',23),('U',47))` A<B [B] **−8.2e-13**：低於 m53 accept 門檻
  1e-12、mover 門檻 1e-6 的 10⁶ 分之一——數值 tie，非改善。約半數 feasible flip 解回與錨
  cost 逐位同值（LP 在新 cell 找到等價幾何）。

## 4. 預期死點驗證

| 死點假說 | 判定 | 證據 |
|---|---|---|
| (1) 替代 relation 使 cone 空掉 | ✅ **主死因（86.8%）** | 459/529 attempt-1 LP infeasible |
| (2) Area 稅抵消 | ⚠️ 成立但形式修正：**HPWL 稅** | 62 feasible 全 worse/tie；d_hgap +3.2e-4 ≫ d_agap +8e-5 |
| (3) boundary-preplaced 等式鎖死 | ❌ **反證** | diag：15 個 attempt-1 infeasible 拿掉 boundary 等式重解，**0/15 翻 feasible** |

死因 (3) 的反證是本 probe 最有資訊量的結果：cone 空**不是** boundary 等式造成，而是
**其餘 ~3000-4900 個 pair 的 fixed-disjunct 鏈 + envelope/bbox 幾何本身**——單 pair flip
需要連動一整串鄰居的 relation 才可行，正是 M57「39% violated bits 是 separation 鏈死」
先驗的拓撲版；對 flip 而言鏈鎖率更高（87%）。可行的 12% 則付 HPWL 稅或恰好 tie
（機制與 M61 event frames 同構：解鎖單一自由度的收益被其他項吃掉）。

## 5. 誠實範圍

- **Pilot 3 案 × top-24 pairs**（npk 全集 3157/4941/4914 → 覆蓋 ~0.5-0.8%，但按 tightness×
  gradient 排序 = 最有動機的 pair 先測）；RED 是預註冊 pilot gate 判定，非全 pair 掃描。
- **Unit-level slide 粗化**：強制一個 k 於全部成員 pair = 「A 整體滑過 B」，強於 evaluator
  的 pairwise 要求——混合 per-pair 拓撲可能可行而本語意不可行（低估可行域；但 86.8% 鏈鎖
  型 infeasible 主要來自 pair 外的 fixed-disjunct 鏈，非本粗化）。
- ladder 只檢查 component 數**增加**；merge 默許（與 M10 精度牆同向、官方 eval 仲裁）。
- Variant B 觸發帶 near-eps=0.2%；diag 樣本每案 ≤5。
- l2base（min(port32, l2stack) 誠實基準）未跑：0 movers 下 union gain 對任何基準恆 0。

## 6. 復現

```powershell
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m64_flip_probe.py selfcheck   # 佈線 gate
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m64_flip_probe.py pilot       # ~7 分（cache 續跑）
& "C:\Users\Nordra\.conda\envs\iccadv\python.exe" m64_flip_probe.py report      # 聚合 + gate 判定
```

檔案：`m64_flip_probe.py`（probe，永不 ship）、`m64_cache.pkl`（550 entries，sig=anchor md5）、
`m64_pilot_stdout.txt`（逐 flip log）。模式 `heavy`/`full`/`l2base` 已實作備而未用（gate 未過）。
