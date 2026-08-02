# M79 — 擴池第一批離線篩選

> # 🚨🚨 2026-08-02 更正：本報告的主結論很可能是 RED，因為基準用錯了
> #
> **實際上傳的 `C:\Users\.01\Downloads\cadc1075\op_wrapper.py`（md5 `6f1f31a2…`，2026-07-29）
> 是 M71 版**：`_M71_ENV = {"ICCAD_CLUSTER_BND_EXPOSE":"1", "ICCAD_CLUSTER_BND_EDGE_PACK":"1"}`
> **以 per-profile env overlay 套用到池中的每一隻 profile、預設 ON**。
> 實測（M80，官方指令跑上傳檔本身）**local total = 1.305389893450635**，
> **不是** `results_shipped_m51.json` 的 1.326473104916827——那份錨是 M71 之前的，已被取代。
>
> **本報告 §3 的「ALL 8 added → 1.305151」幾乎等於出貨的 1.305390。**
> ⇒ 我在 §4 稱為「休眠、沒有任何出貨 profile 在用」的那組旋鈕，
> **早就隨 M71 全域出貨了**；M79 量到的 +0.96~1.14% 是**已經入袋的錢**，不是新的 headroom。
> 我讀到工作樹那段「NO shipped profile uses them → C++ capability was dormant」註解時，
> 沒有意識到它講的是 **profile 層級**的使用，而 M71 走的是**全域 overlay**——不同機制、註解沒涵蓋。
>
> **還沒重做的事**：在 M71-ON 的正確基準上重篩（`audit_cache.pkl` 是 M71-OFF：
> 它的 full-41 = 1.324846 = M71 前的 quality-best）。在那之前，
> **§3 的表與 §6b 的「收斂」敘述都不可引用**。§1（源碼雜湊）、§2（三閘）、§6b 的
> 修正 1–4（M76 的三項條件）不受影響、仍然有效。

## （以下為 2026-08-01 原文，除上方警語外未改）

## 原標題：四隻過 bar 20 倍以上，且 proxy 真的選得到（篩子 GREEN／判定未下）

日期：2026-08-01。執行模式：**純快取分析**——不跑 C++、不跑官方 eval、不產生任何 wall-clock 數字、不碰 `_PROFILES`、不碰送件形。
路線 🅒 的步驟 1–2（`HANDOFF_2026-08-02.md` §3🅒）。步驟 3（48c 投影）、4（真實 dt）、5（OOS）**未做**。

## 一句話

出貨的 `constructive.cpp` 裡有一組**已實作但沒有任何出貨 profile 在用**的休眠旋鈕；
把它們當**新 profile 加進池子**（不是全域套用），四隻各拿 **oracle 0.96–1.14%**，
而且 **realizable ≈ oracle（差 0.001%）** ⇒ 部署用的 `_RH=1.4` proxy 真的會選中它們，
不是 M56/M63 那種「上界很大但選不到」的海市蜃樓。bar 是 0.05% ⇒ **過了 20 倍以上**。

**但這是篩子不是判定**：基準是 K=12 counterfactual、純 in-sample、官方分未投影。見 §4。

---

## 1. 前置：基準釘死（踩過五次的坑）

交接文件鐵律 1 要求基準永遠是真正上傳的那份。逐一驗過：

| 檔 | sha256 | 結論 |
|---|---|---|
| `C:\Users\.01\Downloads\cadc1075\constructive.cpp` | `f663264e…` | 出貨源碼 |
| `teammate_m71_screen/constructive.cpp` | `f663264e…` | **= 出貨源碼** ✅ |
| `teammate_m71_baseline/constructive.cpp` | `cf355f1b…` | ❌ **不是**出貨源碼 |
| `teammate_m29_free_aspect/constructive.cpp` | `f0b89f57…` | ❌ |
| `teammate_m71_final_bnd-*/constructive.cpp` | 各異 | ❌ |

⚠️ **兩個反直覺的點，寫下來免得下次再查一遍**：

1. `teammate_m71_screen` 的 git 顯示 `M constructive.cpp`（+291 行）——**那不是髒工作區，是 HEAD 比出貨版舊**。
   工作區那份就是出貨版。codex 交接文件說它是出貨版 = **正確**。
2. 名字叫 `teammate_m71_baseline` 的那棵樹 git 是乾淨的，但**它的源碼不是出貨版**。名字會騙人。

池子組成也確認過：`_PROFILES` 前 41 隻 = 出貨版逐位相同；41 之後是 `_M55_EXTRA`（41–44）與
`_M73_ESCAPE`（45–48），**預設 gated off**，`_pool_indices()` 在無 env 時逐位等於出貨。
`audit_cache.pkl` sig 吻合本樹全 50 隻（49 + OM16），涵蓋 100 案 ⇒ **步驟 1 確認：快取可用**。

## 2. Sanity 鏈（三閘全綠）

```
G1 audit cache 覆蓋 100 案 x 50 profiles                                    [PASS]
G2 proxy(_RH=1.4) over 出貨 41 隻 = 1.324846                                [PASS]
G3 oracle-min over 出貨 41 隻 = 1.324829；proxy 洩漏 +0.0013%               [PASS]
```

⚠️ **順帶更正一條過期文件**：`rf_score_model.py` 的說明寫「full-pool RF=1.0 重現 **1.3269** 為 sanity gate」
——那是 **M51 之前的 40 隻池**的值。M51 加入 #40 `fa22_fc_pin_tight_wire` 之後，full 41 隻是 **1.3248**
（CLAUDE.md「`ICCAD_ADAPTIVE_POOL=0` 還原 full 41-prof = quality-best 1.3248」才是對的）。
本次實測 **1.324846** 吻合後者。任何人拿 1.3269 當閘門會誤判。

G3 的 +0.0013% 洩漏同時再確認 M31 的「oracle-min == proxy，零漏分」在 M51 池下仍近似成立。

## 3. 主結果

基準 = 出貨 41 隻的 proxy 選擇總分 **1.324846**（K=12）。

| k | 家族 | oracle 增益 | **realizable（proxy 真選）** | 贏案 | top-1 佔比 | dt ≤ 現任最慢者 | 最壞 dt/max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 41 | M55 休眠旋鈕 | 0.960% | 0.959% | 22 | **42%** ⚠️ | 97/100 | 1.12× |
| **42** | M55 休眠旋鈕 | **1.133%** | **1.133%** | 27 | **22%** ✅ | 96/100 | 1.18× |
| 43 | M55 休眠旋鈕 | 1.144% | 1.144% | 20 | 35% | 95/100 | 1.14× |
| 44 | M55 休眠旋鈕 | 1.118% | 1.118% | 23 | 36% | 95/100 | 1.13× |
| 45 | M73 逃生口複本 | 0.000% | 0.000% | 0 | — | 93/100 | 1.07× |
| 46 | M73 逃生口複本 | 0.000% | 0.000% | 0 | — | 98/100 | 1.03× |
| 47 | M73 逃生口複本 | 0.000% | 0.000% | 0 | — | 92/100 | 1.03× |
| 48 | M73 逃生口複本 | 0.000% | 0.000% | 0 | — | 98/100 | 1.03× |

**八隻全加：oracle +1.488% ／ realizable +1.487% → 1.305151**

三個要注意的訊號：

1. **realizable ≈ oracle（差 0.001%）** = 這條路和 M56（winner 不可預測）、M63（可達但恆付稅）、
   M61（proxy 判優官方判劣）**不同類**。增益不需要任何新的選擇機制，現行 proxy 就吃得到。
2. **M73 逃生口複本恰好 0.000%** —— 與隊友 M76 判的 RED **獨立一致**（它們是既有 profile 的複本，
   對 oracle-min 結構上不可能有貢獻）。這同時是本篩子的**反證對照**：它不會亂生增益。
3. **#41 的 top-1 佔比 42%** 超過交接文件 §5 的「最大單案佔增益 < 40%」衛生線 ⇒ **#41 是四隻裡最可疑的**；
   **#42 的 22% 最乾淨**，若只能挑一隻，從 #42 開始。

## 4. 機制：為什麼這裡還有肉

`_M55_EXTRA` 用的六個旋鈕（`CLUSTER_BND_EXPOSE` / `CORNER` / `PERMUTE` / `EDGE_PACK`、
`ANCHORED_BND_REPACK`、`HPWL_SAFE_CLUSTER_SLIDE`）**在出貨的 constructive.cpp 裡已經完整實作**
（宣告 120-133、parse 1938-1943），但**沒有任何一隻出貨 profile 使用它們** ⇒ C++ 能力是休眠的。

`optimizer_constructive.py` 的註解記著 2026-07-29 量過**全域套用**版（41 隻全帶旋鈕）：
`EXPOSE+EDGE_PACK` 拿到 1.305390，**但 17/100 案退步**，所以沒有採用。

**差別就在這裡**：全域套用會改動既有 profile 的行為（可能變差）；
**當成新 profile 加進池子是下檔保護的**——proxy 不選它就完全不傷分，只花 runtime。
而 48 核下「只要不比現任最慢者更慢，加它就是免費的」（交接文件 §3🅒 的 free-restore 預算：
`c* = 22.2~26.8` vs 48 核 ⇒ 池子可長到 75–80 隻）。

同時這也**滿足 🅒 的關鍵限制**：這是「加**新** profile」（新的排版策略組合），
不是「加回**舊**的」（tier-5 已經吃完，加回舊的在大案帶是 0 隻，見 M67-F 更正 C）。

## 5. 誠實範圍（這是篩子，不是判定）

1. **🚨 基準是 K=12 counterfactual。** `audit_cache.pkl` 沒有 `_band_env` overlay，
   但出貨跑的是 K=4（n>100）／K=8（60<n≤100）。**M67-F 更正 B 就是栽在這件事上**
   （tier-3 常數在出貨組態下失效）。⇒ **任何存活者必須在出貨 K overlay 下重驗**，
   而 `audit_cache_kband.pkl` **至今沒有取樣過**（`profile_audit.py --kband` 已寫好、未執行）。
2. **純 in-sample。** M75 的教訓是 in-sample 會給**相反符號**（in-set +0.0104% / OOS −0.0111%），
   M67-D 量到 adaptive 切法的 OOS 稅是 in-set 的 **27 倍**。交接文件 §5 第 5 條要求 OOS 且判準事前註冊。**未做。**
3. **官方分未投影。** 上表全是 **RF=1.0 的品質數字**。48 核官方分要跑 `m67e_rf48.py project`（步驟 3），
   **未做**。dt 有 3–5 案會把牆抬到 1.12–1.18× ⇒ 不是全免費，要付 RF 稅。
4. **dt 來自快取、不是重量。** 「dt ≤ 現任最慢者」是用 `audit_cache` 的 dt 算的（同樣 K=12）。
   步驟 4 的真實 dt 量測是 🔴 車道，**且交接文件要求等隊友的 M74 落地**（改到同一個池子）。
5. **`_M55_EXTRA` 四隻的實際 env 定義**見 `optimizer_constructive.py` 的 `_M55_EXTRA`；本報告只用索引指稱，
   沒有在此複述以免抄錯。

## 6. 下一步（依序）

1. **`m67e_rf48.py project`** 把四隻（或只 #42）投影成 48 核官方分 —— 🟢 快取算術，最便宜的下一步
2. **出貨 K overlay 重驗**：跑 `profile_audit.py --kband` 取樣 → 在 K=4/K=8 下重算本表
   （這同時補上 M67-F 更正 B 留下的那個洞）
3. **OOS 驗證**（`m67_oos_probe.py` 協定，判準事前註冊）—— §5 硬性要求
4. 真實 dt vs 現任最慢者 —— 🔴，**等 M74 落地**
5. 過了才談 ship 形（env-gated、預設 off、全鏈重驗）

⚠️ **本輪不動工、送件形零改動。**

## 6b. 🚨 M76 覆蓋範圍回覆的交叉檢查（2026-08-01，隊友 `M76_TIER5_COVERAGE_NOTE.md`）

隊友從 escape-hatch 那一側回覆了「往池子加新 profile 的空間」——**結論與 M79 從相反方向撞在一起**，
但也修正了 M79 的基準與條件。逐條記錄。

### 收斂（兩邊獨立得到同一句話）

隊友 §結論-1~3：「**沒有『還原被砍的』空間了**——48 核上池子已是 35/41、每帶相同，剩的 6 隻是必須留的巨獸；
**要加只能是新造的異質候選**；M76 的 full-union 天花板 **−0.388%** 證明異質候選確實還有肉，
只是 41 隻 knob-off 副本買不起（ΔRF +57%）——**天花板存在，但要挑便宜的路徑上去**。」

M79 找到的正是那條便宜路徑：四隻異質候選、**dt ≤ 現任最慢者 95–97/100 案**、realizable ≈ oracle。
兩邊沒有互相參考。另外隊友 §肯定-8（proxy = per-case oracle ceiling；full-union 混合池 −0.388%
**贏過** 2-way oracle −0.340%）**獨立佐證** M79 的 realizable ≈ oracle 不是巧合。

### 修正 1 —— M79 的 mid 帶基準是錯的（但錯在保守方向）

隊友 §肯定-5 在**他們的樹（M74 現況）**實測 `_pool_indices()`：48 核下**每一帶都是 35 隻**。
機制是三個 tier 同時失效，其中 **M74 把 tier-3 從 universal 降級成 `_effective_cores() <= _M45_MID_CORES_MAX (=16)`**。

**我們這棵樹沒有那個降級**（`_M45_MID_CORES_MAX` 不存在；`m67e_rf48.pool_shipped()` 對
`_M45_BAND_DROP` 是**無條件**套用）⇒ M79 算 max-setter 時，(60,100] 帶用的現任池是 **26 隻**，
而 M74 之後應是 **35 隻**。池子越大 → max-setter 越慢 → **M79 的「dt ≤ 現任最慢者」在 mid 帶偏嚴**
⇒ 表中 95–97/100 是**下界**，真值只會更好。

但方向對不代表基準對。⇒ **§5 誠實範圍第 1 條升級為硬性前置**：
這正是交接文件 §2 協議第 3 點（「M74 落地後，我們用新基準重跑一次篩選才下結論」）**具體被觸發**。

### 修正 2 —— in-sample 挑選的 OOS 轉移率 ≈ 5%（第四次量到）

隊友 §否定-5 的乾淨對照（`m73x` vs `m73big`：同機制、同 gate、只換來源集）：

| | in-set 100 | OOS@16c | OOS@48c |
|---|---|---|---|
| `m73x` − `m73big` | **+0.127pp** | **+0.006pp** | **+0.006pp** |

若 M79 的 +1.13% 以 5% 轉移，OOS 只剩 **+0.057%** —— **貼著 0.05% bar**。

⚠️ 但有一個結構區別值得註冊成**假說（未驗，不可拿來當免死金牌）**：
`m73x`/`m73big` 的差別只是「**複製哪幾隻既有 profile**」= 在近乎同質的家族內**重新挑子集**
——那正是 M55/M56/M75 反覆判死的東西。M79 的候選是**啟動從未被任何 profile 使用過的休眠 C++ 路徑**
= 加**新機制**。而 M67-D 量到 **portfolio 增益本身 OOS 轉移良好**（in-set 10.22% → OOS **11.50%**，比 in-set 更好），
不轉移的是**子集挑選**。⇒ 假說：**「加機制」的轉移率 ≠「挑子集」的轉移率**。
**這個假說必須用 OOS 實測判，判準事前註冊、不得事後移動。**

### 修正 3 —— 免費預算是個位數，且與 tier-5 共用同一個賭注

- 隊友 §推測-2：偵測核數是有效核的**上界**（本機比值 0.63）⇒ 48 偵測 ≈ **30 有效**；
  重帶 `c* = 23.4` ⇒ 餘裕只剩 **~6 隻**。M79 提 4 隻 → **塞得下，但沒有多少餘裕**。
- 隊友 §推測-4（**最大的條件依賴**）：若評分機**有效並行度 < 40**，tier-5 不觸發 → 重帶池回到 13
  → 但**同時「加 profile 免費」整個作廢**（wall 翻成 `Σ/cores` bound，12 核實測 dRF **+4.5%**）。
  ⇒ **M79 的價值與 tier-5 共用同一個賭注**；賭輸的話 M79 不是變小，是**變號**。
- 隊友 §否定-4 也直接點名：免費結論**限 `dt <= 當前 max-setter`**。M79 表中最壞 1.12–1.18×
  的那 3–5 案**就是要付錢的那些**，不能當成免費。

### 修正 4 —— OOS 必須用 48 核形狀量

隊友 §否定-6：in-set 100 在 16 核／48 核形狀下**逐位相同**，OOS 卻差 **2.7 倍**（+0.294% → +0.107%）。
⇒ 任何與 adaptive tier 有交互的機制，OOS 一律要 `m67_oos_probe.py --force-cores 48`。
⚠️ **我們樹上的 `m67_oos_probe.py` 是否有 `--force-cores` 未確認**（隊友那支有）。

### 工具重疊 + 編號衝突（要跟隊友講）

1. 隊友 §結論-4 有一支 **`m77_ml_candidate_probe.py`**：吃任何 optimizer 的官方 results json，
   當第 42 隻候選接進池子走 proxy 仲裁，直接吐 **portfolio delta（bar 0.05%）+ dRF@48c**
   ——**和 M79 是同一個問題**，只是入口是 results json 而非快取索引。
   `m77_ml_candidate_probe.py` / `M77_ML_GATE_NOTE.md` / `M76_REPORT.md` / `m76_escape_probe.py`
   **都不在我們樹上**（在他們機器）。⇒ 應該跟他們要，而不是各做一支。
2. 🚨 **編號撞號**：隊友的 **M77 = ML candidate gate**；我方的 **M77 = LP 時間預測器**
   （`m77_lp_time_predictor.py` + `M77_REPORT.md`，在 `teammate_m29_free_aspect/`）。
   ledger 是以編號索引的，兩個 M77 會出事。**建議把我方那支改編號 M80**（M78 閒置 gate、M79 擴池不衝突）。

### M79 判定的修訂

原「篩子 GREEN／判定未下」**維持**，但前置條件明確化為三項且**全部未滿足**：
**(a) 等 M74 落地後用 35 隻的 mid 帶基準重篩**、**(b) OOS 用 `--force-cores 48` 實測、判準事前註冊**、
**(c) 認清它與 tier-5 共用同一個賭注**。在這三項之前，+1.13% 不可對外引用為可實現增益。

---

## 7. 檔案 / 復現

| 檔 | 說明 |
|---|---|
| `m79_pool_expand.py` | 三個 mode（`gate0` / `screen` / `report`），永不 ship |
| `results_M79_pool_expand.json` | 本表全部數值 |
| `m67e_cache.pkl` | 已寫入本次新算的 800+ 筆逐 (case,profile) 官方 cost（lazy cache，重跑會快很多）|

```bash
cd C:/ICCAD_ml/teammate_m71_screen
"C:/Users/.01/anaconda3/envs/floorset/python.exe" m79_pool_expand.py gate0    # 三閘
"C:/Users/.01/anaconda3/envs/floorset/python.exe" m79_pool_expand.py report   # 全套（首跑 445s，之後快）
```

依賴：`audit_cache.pkl`（sig 須吻合本樹 `_PROFILES`）、`m67e_rf48.py`、`m67e_cache.pkl`。
未動任何 shipped 檔、未改 `_PROFILES`、未產生新的 wall-clock 量測。
