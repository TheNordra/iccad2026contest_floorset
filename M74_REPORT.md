# M74 — adaptive stack 常數 regen（2026-07-30）

**判定：GREEN，已落地。** 範圍 Final-only，已上傳的 M73 Beta 包（`op_wrapper.py` md5 `c2e27c99…`）零改動。

---

## 1. 起因

M42/M45/M49/M50 的全部 drop 常數都是從 `audit_cache.pkl` 推的，而那顆 cache 是 **2026-07-10** 產的：

| 面向 | cache 內容 | 出貨組態 |
|---|---|---|
| binary | 07-10 pre-M71 exe | 07-29 M71 exe |
| M71 旗標 | 無 | 每隻 profile 都開 |
| REFINE | 一律 12 | n>100 K=4；60<n≤100 K=8 |

⇒ M45 的 strict selection-preserving gate **從未在出貨組態下成立過**（M67-F correction B）。而且更嚴重的是：`profile_audit.run_one()` 與 `m49_refine_probe.run_case()` 都沒有套 `_m71_env()`，**所有離線 gate 一直在量 pre-M71 的 placer**——因為 C++ 旗標預設 OFF，binary 單獨逐位不變，所以 gate 全綠卻量錯對象。

根因是簽章：六個消費者只檢 `repr(_PROFILES[:_M55_BASE_LEN])`，**不含 exe md5、不含 overlay**。

---

## 2. 結果

### In-set（本地 100 案，RF=1.0）

| 組態 | M73 | **M74** | delta |
|---|---|---|---|
| 預設（≤16 核） | 1.305390 | **1.293461** | **−0.769%** |
| `ADAPTIVE_CORES=48`（評分機） | 1.295548 | **1.293461** | **−0.158%** |
| `ADAPTIVE_POOL=0` 天花板 | 1.2929 | 1.2929 | — |
| **adaptive 品質稅** | **+0.967%** | **+0.046%** | |
| avg runtime | 1.52s | 1.45s | |

逐案：**14 movers，14 個全部變好、0 個退步**（case 89 −9.3%、85 −4.5%、87 −3.8%、91 −2.6%、79 −5.2%、68 −8.4%）。100/100 feasible。

### OOS（240 案 held-out，同 M67-D/M72 語料）

| band | 舊 @16c | M74 @≤16c | **M74 @48c 形狀** |
|---|---|---|---|
| n≤60 | 1.707104 | +0.000% | +0.000% |
| 60<n≤100 | 1.542581 | +0.120% | **−0.583%**（25 好 / 5 壞） |
| n>100 | 1.595348 | −0.776% | −2.238%（46 好 / 0 壞）※ |
| **ALL** | 1.590441 | −0.686% | **−2.068%**（71 好 / 5 壞） |

※ **誠實範圍**：n>100 那欄的 −2.238% 是對「舊常數 @16 核（tier-5 未觸發）」比的。在真 48 核上，**舊常數也會由 tier-5 把整組 `_BIG_REDUNDANT_IDX` 還原**（m67g V2 逐核驗過），跑的是同一個 35 隻 pool ⇒ **重帶在評分機上新舊相同**。重帶 regen 的價值在 <40 核機器與常數本身的正確性，不在 48c 分數。
⇒ **48c 評分機真正的 OOS 增益 = mid 帶的 −0.583%**，全部來自 tier-3 降級（見 §3.3）。

---

## 3. 落地的改動

### 3.1 `_BIG_REDUNDANT_IDX`（M42）
仍是 22 隻、T=100 仍 all-win，但**成員大幅改變**：`{1,3,6,18,20}` 出、`{8,12,13,21,40}` 進。模型 local RF=1.0 1.2935，與 M41 post-swap baseline 逐位相同。

### 3.2 `_M45_LOWCORE_DROP`（tier-4）
三個帶全部重推，全部 strict all-equal：
`(40,60]` 24 隻（wall @4c 4.57→1.69s）、`(100,110]` 5 隻、`(110,inf]` 6 隻。

### 3.3 `_M45_BAND_DROP`（tier-3）— 內容重推 **且降級為 cores-gated**
- 內容：9 → **15** 隻，strict all-equal 成立，mid wall @12c 2.21→1.44s（−35%）。
- **但 OOS 打臉**：同 80 案 held-out mid 帶，**跑滿 pool 比這個剪法好 −0.702%（30 好 / 0 壞）**。in-sample 嚴格相等完全沒有轉移——正是 M67-D/M55/M72 的 doctrine。
- 而它在高核上幾乎買不到 wall：48 核時 mid 帶是 max-setter-bound（實測 c\* max **15.2**），wall 只從 1.32 → 1.30s，`rf_score_model` 投影 **+0.00%**。
- ⇒ 新增 **`_M45_MID_CORES_MAX = 16`**，tier-3 只在 `_effective_cores() <= 16` 才開。門檻取自 c\* max 15.2；偵測核是有效核的**上界**（本機 16 邏輯核 ≈10 有效），所以報 16 的機器仍是 sum-bound、仍該剪。誤偵測方向：`_effective_cores()` unknown→9999 ⇒ tier **關**＝滿 pool＝品質安全側（沿用 tier-4 的 fail-open 慣例）。

> 這與 tier-5 是同一個道理的鏡像：高核下 pool 剪法停止買 wall、只剩付品質。

### 3.4 `_M49_REFINE_BAND` mid：K=8 → **6**
重掃 K∈{4,6,8,10}×{big,mid}：
- **big K=4 現在是純贏**，不再是品質交換：weighted local **−0.056%**（movers 87、94，**兩個都變好**；M49 當年是 +0.027% 且只有 case 85 動）。band wall @12c −53%，20/20 median-independent WIN。**M71 把符號翻了**。
- **mid K=6 在每一格投影上弱優於 K=8**（@4c M=6 −2.13% vs −1.63%；worst cell 都是 +0.02%），local 品質同級（+0.019% vs +0.018%），因為多砍 wall（−31.0% vs −25.0%）。OOS 也證實 K=6 略優於 K=8（+0.120% vs +0.131%）⇒ **K 不是 mid OOS 退步的原因，tier-3 才是**。
- mid K=4 ungated 仍不過 bar（local +0.049%、worst cell +0.05%），續留低核 tier。

---

## 4. 基礎建設修正（本次真正的長期價值）

| 檔案 | 修正 |
|---|---|
| `profile_audit.py` | 新增 `base`\|`ship` 模式（`audit_cache.pkl` / `audit_cache_ship.pkl`）；套 `_m71_env()`，ship 再套 `_band_env(n)`，**順序與 wrapper `:1058-1062` 完全一致**；child env 先剝 `ICCAD_*`；釘 `ICCAD_ADAPTIVE_CORES=12`；ship 完成後自動 cross-check「無 overlay 的 40 案必須與 base 逐位相同」 |
| 簽章（6 個消費者） | `repr((repr(PROFILES), MODE_KEY, exe_md5))`，`MODE_KEY` 在 ship 模式**還含 `_M49_REFINE_BAND`/`_M50_REFINE_LOWCORE`/`_M45_CORES_MAX`** ⇒ 重編 exe、改 M71 overlay、retune K 都會讓 cache 失效 |
| `m49_refine_probe.py` | `run_case()` 補 `_m71_env()`（**這是「gate 全綠卻量錯 binary」的根因**）；`pool_at()` 跟上 cores-gated tier-3；`trace` 模式加註 `EXE46` 是 pre-M71 ⇒ 位置不匹配屬預期 |
| `rf_score_model.py` | 改讀 ship cache；四個 drift assert 加 `ICCAD_REGEN=1` 逃生門（降級成 warning），否則第一個 assert 會在印出 M45 建議前就 abort |
| `m67_oos_probe.py` | `_sig()` 現在釘**全部** adaptive 常數（快取存的是整個 `opt.solve()` 輸出，舊 key 只有 binary+`_PROFILES` ⇒ 換常數會靜默重用舊結果）；gate0 的 `len(_PROFILES)==41` 改錨 `_M55_BASE_LEN`（M72 之後那是 45，一直誤報 FAIL）；tier-3 相關斷言改成 cores-aware |
| `m67g_tier5_gate.py` | V1 kill-switch 原本要求「tier-5 關掉後 48c 與 12c 每個 n 都相同」——tier-3 cores-gated 之後這前提不成立（n=61-66 合法不同）。改成 (a) 兩個都在 tier-3 門檻之上的核數必須全等（隔離 tier-5），(b) n>100 仍與 12c 比 |
| `m54` / `m56` / `m67e` | pool 鏡像跟上 cores-gated tier-3（`m67e` 尤其重要：它就是 48c 投影工具） |

---

## 5. 驗證

- `regression_suite.py` **7/7 ALL PASS**（762s）
- 兩個 m49 gate 的 **control K=12 vs cache 都是 EXACT 100%** ⇒ 新 base cache 與現行 exe+M71 完全一致
- base cache 的 full-pool total **1.2929**，獨立重現官方 `ADAPTIVE_POOL=0` 錨
- ship cache cross-check：40 個無 overlay 案逐位相同
- 官方 eval：預設 1.293461 / 48c 1.293461 / POOL=0 1.2929，皆 100/100 feasible
- 核數掃描：≤16 tier-3 開、≥17 關、≥40 tier-5 開、unknown→滿 pool（品質安全側）

### ⚠️ Bash 工具坑
`regression_suite.py` 走 **Bash 工具**時 `m48` 必 FAIL（`exe=False`、`recompiled 32B -> 32B`）——sandbox 擋 `.exe` 寫入，**與程式碼無關**。同一個指令走 PowerShell 是 ALL PASS。**送件前的 gate 一律用 PowerShell 跑。**

---

## 6. 錨檔

- `results_shipped_m71.json` = **已上傳 Beta tar 的身分錨，本次未動**。`make_submission.py:64` 的 `_ANCHOR` 仍指著它 ⇒ 現在跑 `verify` 會報 diff，**這是預期**，不是壞掉。真要為 Final 出貨時才改指。
- 新錨：`results_M74_default.json`（1.293461）、`results_M74_cores48.json`（1.293461）、`results_M74_pool0.json`（1.2929，**首次有 json 錨住這個天花板**）。
- OOS cache 快照：`m67_oos_cache.pkl.preM74`（舊常數基準）、`.M74k6`、`.M74k8`、`.M74notier3`、`.M74_48shape`。
- `audit_cache.pkl.preM74` = 舊的 pre-M71 cache 備份。

---

## 7. 留給 Final 的事

1. **要出貨就得重跑換件鏈**：`regression_suite.py`（PowerShell）→ `make_submission.py all` → `m67c_make_linux_bundle.py` + GPU 機 WSL `verify_final_tar.sh`（含 `final48`）→ Drive 覆蓋；並把 `_ANCHOR` 改指 `results_M74_default.json`、`m67c` 的 48c 錨改 `results_M74_cores48.json`。
2. **`_M45_MID_CORES_MAX = 16` 的殘餘賭注**：若 Final 機器回報 17-24 核但有效並行度其實 <15，tier-3 會關掉而 mid 帶付 sum-bound 的 wall。48 核已知安全（關掉是對的），≤16 也安全。
3. **未量**：mid 帶在 M74 常數下的 48c wall 代價（§3.3 用的是 pre-M74 的 audit dt 推的 c\*）。
4. `m55_dropset_cv.py` / `m56_percase_oracle.py` 是 RED 存檔 probe，簽章已跟上但**沒有在 M74 下重跑**，其結論仍是 pre-M71 的。
