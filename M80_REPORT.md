# M80 — 出貨版 runtime 乾淨重量 + 給 M74 的 RF 彈性：**完成**；附兩項基準更正

日期：2026-08-02（夜間獨占視窗）。路線 🅐 步驟 1–3。
量測對象 = **真正上傳的** `C:\Users\.01\Downloads\cadc1075\op_wrapper.py`（md5 `6f1f31a2…`，鐵律 1）。
送件形零改動、`_PROFILES` 未碰。

## 一句話

20 次官方指令重跑、**每次 quality 逐位相同**，逐案取最小值：這台機器的出貨版 avg **1.0628s**
（記錄裡舊機 1.52s ⇒ **0.70×**）。給隊友 M74 的答案是**彈性 0.176–0.300**
⇒ 1.52s→1.45s 那個 −4.6% 值 **−0.81% ~ −1.38% 官方分**（s∈[1.5,2.5]）。
過程中撞到兩件更重要的事：**出貨基準其實是 1.305390 不是 1.326473**，以及 **M79 因此 RED**。

---

## 1. 🚨 基準更正：出貨的是 M71，local total = 1.305390

照鐵律 1 拿真正上傳的檔案跑官方指令：

```
local total = 1.305389893450635      (results_shipped_m51.json = 1.326473104916827)
```

原因在上傳檔 line 555：

```python
_M71_ENV: Dict[str, str] = {"ICCAD_CLUSTER_BND_EXPOSE": "1",
                            "ICCAD_CLUSTER_BND_EDGE_PACK": "1"}
```

= **M71，per-profile env overlay 套到池中每一隻 profile、預設 ON**（2026-07-29 出貨）。

**獨立交叉驗證**（兩條完全不同的路徑撞出同一個數字）：

```
M80  官方指令實跑上傳檔                        → 1.305389893450635
rf_score_model.py --m71（讀 audit_cache_m71）  → local RF=1.0 total shipped = 1.3054
```

⇒ `audit_cache_m71.pkl` 就是**正確基準**，`audit_cache.pkl`（full-41 = 1.324846）是 pre-M71 語意。

**過期的地方**：`HANDOFF_2026-08-02.md` §0 表格「已上傳 cadc1075 = 1.3265」、
`CLAUDE.md` 現況一句話「shipped 1.3265」。兩處已加更正註記。
`results_shipped_m51.json` 仍可用，但它擋的是 **pre-M71 逐位**，不是出貨逐位。

⚠️ 這是鐵律 1 的**第六次**，方向相反：不是拿錯檔案，是**文件沒跟上出貨**。

## 2. 🚨 連帶：M79 = RED

| 池 | RF=1.0 total |
|---|---:|
| M71-**OFF** full-41（= M79 的基準）| 1.324846 |
| M79「8 隻全加」投影 | **1.305151** |
| **實際出貨（M71-ON + adaptive 切）** | **1.305390** |
| **M71-ON full-41** | **1.2929** |

M79 量到的 +1.13% 幾乎逐位等於 M71 已經入袋的錢；而 M71-ON 的 full 池（1.2929）
**比 M79 的投影還好 0.9%** ⇒ M79 測的是一個**比已出貨版本更差**的東西。**RED。**

死因是我讀錯註解：工作樹寫「NO shipped profile uses them → C++ capability was dormant」
講的是 **profile 層級**的使用，M71 走的是**全域 overlay**——不同機制，註解沒涵蓋。
`M79_REPORT.md` 開頭已加警語，§1（雜湊）/§2（三閘）/§6b（M76 三項修正）不受影響。

## 3. 主結果：乾淨 runtime

**方法**：完美閒置視窗拿不到（M78 兩次 FAIL：機器上有使用者活動 + Claude harness 自己關不掉），
所以改變**估計量**而非門檻——重跑 N 次取**逐案最小值**（競爭只會增加牆鐘時間 ⇒ min 是一致估計量）。
每次重複另記逐程序 CPU 增量當共變量。

**硬 gate（每次都跑）**：quality 必須與 repeat 0 逐位相同；stderr 出現 fallback 立即中止。

```
20 repeats, wall 134-148s each, quality bit-exact 20/20
avg_runtime per repeat: 1.1139 .. 1.2540 (p50 ~1.157)
背景負載: 每次 48-83% of one core（全是 Claude harness 本身）
```

**逐案最小值**：

| | |
|---|---:|
| mean | **1.0628s** |
| weighted mean（e^(n/12)）| **1.4895s** |
| p50 / p90 / max | 1.0290 / 1.8183 / 2.5843s |
| 跨重複的 max/min 離散度 | p50 **1.23×** / p90 1.47× / 最壞 **2.16×** |

**紀錄裡舊機 avg 1.52s ⇒ 這台 0.70×（快約 30%）**，而且是在 M71 做**更多**工作的情況下。

最重的案子（逐案 min）：78(n=99) 2.584s、77 2.355s、72 2.163s、71 2.113s、73 1.992s、
67 1.872s、**99(n=120) 1.858s**、70 1.848s、79 1.831s。

⚠️ **1.23× 的逐案離散度本身就是結論**：這台機器在有使用者活動時，單次量測的逐案 runtime
不可信到 20% 量級 —— 這正是 M77 那個「兩份快取差 1.27–1.46×」的同一個現象。

## 4. 給隊友的交付：RF 彈性

被 floor 壓住的案子多花時間不用付錢、省時間也拿不到錢，所以只有**沒觸底的權重**會反應：

```
d ln(官方分) / d ln(我們的 runtime) = 0.3 × (未觸底的 cost 權重佔比)
```

以 alpha 校準（`M_i = 3.161·t_i^alpha`，`Downloads/cadc1075_results.json`）+ 本次實測 t：

| s（grader 秒／本機秒）| 加權 RF | 觸底權重 | **彈性** | **−4.6% runtime 值多少** |
|---:|---:|---:|---:|---:|
| 1.0 | 0.7159 | 73.5% | 0.0794 | −0.365% |
| **1.5** | 0.7541 | 41.4% | **0.1758** | **−0.809%** |
| **2.0** | 0.8035 | 15.3% | **0.2541** | **−1.169%** |
| 2.5 | 0.8567 | 0.0% | 0.3000 | −1.380% |
| 3.0 | 0.9049 | 0.0% | 0.3000 | −1.380% |

（負值 = 變好，官方分低者為佳。）

**⇒ M74 的 1.52s→1.45s 在 s∈[1.5,2.5] 值 −0.81% ~ −1.38% 官方分。**

⚠️ **三項誠實範圍**：
1. **s 不能沿用 M67-E 的 [1.5,1.7]**——那是在**舊機**校準的，而本次證明舊機比這台**慢約 1.4×**
   ⇒ 這台的 s 相應**更大**。要看整條掃描，別挑單一欄。
2. **彈性是一階近似**，假設 runtime 均勻變化；M74 實際改的是特定帶的池，不均勻。
3. **本機 32 邏輯核、grader 48 核**。32 核跑 ~35 隻是輕微超額訂閱（wall = max(max_i, Σ/32)），
   48 核則是純 max-bound ⇒ 本次 t 可能略高於 grader 的對應值（在 s 之外）。

## 5. 附帶發現

### 5a. 🚨 出貨的 adaptive-pool 常數是 pre-M71 推的

`rf_score_model.py --m71` 在 M71 overlay 下重推常數，**三組全部漂移**：

| 常數 | only-in-model | only-in-shipped |
|---|---|---|
| `_BIG_REDUNDANT_IDX` | 8,12,13,21,23,40 | 0,3,6,18,20 |
| tier-3 `_M45_BAND_DROP` | 0,1,5,6,12,14,21,24,31,32 | 2,13,17,20 |
| tier-4 (110,∞] | 2,17,21,40 | 26 |

⇒ **出貨在砍的不是 M71 語意下該砍的那些。** 這是 M67-F 更正 B（K overlay）的同族問題但更大條
（那次只是 K，這次是 M71 本身）。

### 5a-2 🚨 補量完成：**M71 讓 adaptive 切法的品質代價變成 8 倍**

同夜補跑了 `profile_audit.py --m71 --kband`（= **真正的出貨組態**：M71 overlay + 出貨的
K=8(mid)/K=4(n>100) band overlay；cores=32 ⇒ tier-4 正確地不啟用）→ 新快取
`audit_cache_m71_kband.pkl`（**獨立檔案，既有快取零改動**）。這就是 M67-F 更正 B 記著
「取樣本身尚未執行」的那個洞，至今第一次執行。

`profile_audit` 自己的摘要：

```
full (41)          1.2927      prunable ids: [5, 7, 28, 30, 31, 32, 33]   ← 只有 7 隻
pruned (34)        1.2927      ← 逐位相同 = 這 7 隻是真的免費
pruned+om16 (35)   1.2927
```

`rf_score_model.py --m71 --kband`：`SANITY full-pool = 1.2927`、`local RF=1.0 total shipped = 1.3033`。

**接著補了 `m67e_rf48.py` 的 `--m71/--kband` 模式**（純附加、預設 off、寫獨立快取
`m67e_cache_m71_kband.pkl`，key 構造逐字比照 `profile_audit.py`/`rf_score_model.py`）。
兩道自我閘門：預設模式 full-41 = **1.324846 逐位不變**；`--m71 --kband` full-41 = **1.292657**
= `profile_audit` 自報的 1.2927 ✅。

**⇒ 完整的品質代價表**（全部同一基準 `audit_cache_m71_kband.pkl`）：

| 組態 | 大案帶池 | total | vs full-41 |
|---|---:|---:|---:|
| full 41 | 41 | 1.292657 | — |
| **shipped @48c（grader，tier-5 ON）** | **35** | **1.295548** | **+0.224%** |
| shipped @32c（本機，tier-5 OFF）| 13 | **1.305390** | +0.985% |
| 35 everywhere（≈ M74 @48c）| 35 | 1.293463 | +0.062% |

**🚨 `shipped @32c = 1.305390` 逐位等於 M80 的官方指令實測值**
⇒ 整條「M71 + kband 快取 → proxy 選擇 → 官方 cost」的模型鏈**交叉驗證通過**。
同時**解掉**先前記的那個「模型 1.3033 vs 實測 1.305390」落差：1.3033 只是
`rf_score_model` 用了不同的 cores 假設（tier-4 有觸發），不是模型錯。

### ⚠️ 對本節前一版結論的更正

初稿寫「M71 把切法的品質代價從 +0.12% 拉到 **+0.98%**，約 8 倍」——**那個比較用錯了組態**。
+0.98% 是 **tier-5 關閉**（本機 32 核，`_effective_cores_hi()=32 < _M67F_CORES_MIN=40`）下的數字。
**grader 是 48 核、tier-5 會觸發**，大案帶自動回到 35 隻 ⇒ **實際代價只有 +0.224%**。

like-for-like（兩邊都 tier-5 OFF）那個 8 倍**仍然成立**（pre-M71 +0.123% → M71 +0.985%），
但它只描述 **≤39 核**的機器，**不是評分機**。⇒ **對送件分數的實際影響遠小於初稿所寫。**
殘餘的 +0.224% 幾乎全部來自 **tier-3**（中案帶 26 隻 vs 35），而 tier-5 不還原它。

這正是 M67-F 更正 C 記過的同一個陷阱（「θ 是在 12 核 tier-5 關閉下量的，卻套用在 48 核
tier-5 開啟的組態上」）。**任何 pool 數字都必須先講清楚是哪個核數。**

### 🎁 附帶算出隊友要的東西：M74 的 tier-3 降級在 grader 上值多少

M76 note §肯定-5 說 M74 把 tier-3 從 universal 降成 `_effective_cores() <= 16`
⇒ 48 核下 tier-3 也關掉 ⇒ 逐帶都是 35 隻。代進上表：

```
shipped @48c 1.295548  →  35 everywhere 1.293463   =  -0.161% 品質
```

**⇒ M74 的 tier-3 降級在 grader 的 48 核上值 −0.161% 品質**（RF 側代價另計：中案帶 26→35 隻）。

⚠️ **本輪不動任何池常數**——那是隊友 M74 的領域，且任何改動要走完整重驗鏈
（m49 三 gate → regression_suite → make_submission → WSL 逐位）。本節只提供資料。

### 5b. Windows 上跑出貨包會**靜默退回 Python SA**

第一次跑就被咬到，gate 擋下來了。根因：包裡是 `bin/constructive_linux`，Windows 要現場編譯；
編譯鏈寫死的 `C:\msys64\ucrt64\bin\g++.exe` **存在**、`--version` **正常**，
但**產出二進位**需要 `C:\msys64\ucrt64\bin` 在 **PATH** 上（driver 要找 `cc1plus`/`as`/`ld`）
⇒ 沒 PATH 就 0.1 秒 exit 1、**stderr 空白**、smoke 失敗、無聲換成 SA，**eval 照樣吐分數**。
騙人的地方：`-fsyntax-only` 會過、從 Git Bash 跑也會過。**Linux grader 不受影響**。
`m80_rf_remeasure.py` 已內建 `MSYS_BIN` 修正。

### 5c. `m67e_rf48.py` 沒有 M71 模式

它寫死讀 `audit_cache.pkl`（M71-OFF）⇒ **所有 48c RF 投影都建立在 pre-M71 positions 上**。
`rf_score_model.py` 有 `--m71`、`audit_cache_m71.pkl` 也已存在 ⇒ 補這個模式是純快取工作。
影響方向**未評估**。

### 5d. `rf_score_model.py --m71` 尾段會 assert 失敗

`line 652` 的 M46-block `alpha=1 sanity`（0.93047 vs 0.92888 @cores=12）在 M71 overlay 下不成立。
該模式的 docstring 說漂移檢查改成 report-only，但這一個 assert 沒被轉換 ⇒ 工具小洞，
主要輸出（SANITY / 逐帶表 / 常數漂移）在 assert 之前就印完了、不受影響。

## 6. 檔案 / 復現

| 檔 | 說明 |
|---|---|
| `m80_rf_remeasure.py` | 四 mode：`setup` / `run --repeats N` / `report` / `rf`（永不 ship）|
| `results_M80_rf_remeasure.json` | 20 次逐案 runtime + 逐次背景 CPU 共變量 + `per_case_min` |
| `m80_run.log` | 背景執行輸出 |
| `m78_idle_gate.ps1` | 閒置硬 gate（判準事前寫死；本輪兩次皆 FAIL，見 §3 方法）|

```bash
cd C:/ICCAD_ml/teammate_m71_screen
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
"$PY" m80_rf_remeasure.py setup              # 從 Downloads/cadc1075 建乾淨驗證目錄
"$PY" m80_rf_remeasure.py run --repeats 20   # ~45 分（斷點續跑）
"$PY" m80_rf_remeasure.py report             # 逐案最小值
"$PY" m80_rf_remeasure.py rf                 # RF 彈性表
```

未動任何 shipped 檔、未改 `_PROFILES`、未碰 codex 程序。
