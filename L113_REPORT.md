# L113 — 送件樹確認 + route A 落地（2026-08-10）

接 `HANDOFF_2026-08-09.md` §3/§5。做完兩件事：**確認送件樹**、**把 route A 落到那棵樹並修好三個會靜默出錯的地方**。
過程中發現 route A 現況**不能送**——不是分數問題，是會 10.0 或直接掛住。

---

## 1. 送件樹 = 隊友 `m29-free-aspect` 分支（origin tip `2aae61c`）

`git fetch` 後三個遠端分支：`main`（M27 時代，死的）、`m71-mechanism-screen`（= 我方 screen 樹）、
**`m29-free-aspect` `2aae61c`**（頂端三個 commit 全是 M80）。

| 比對 | 結果 |
|---|---|
| `m80_probe/constructive.cpp` vs 該 tip | md5 `f35bd39b` **完全相同** |
| `m80_probe/optimizer_constructive.py` vs 同分支 wrapper | **只差 2 個 hunk、+304/−1 行** = route A |
| 該分支有 `make_submission.py` / `regression_suite.py` | ✅（7 項：m48/rf/m49big/m49mid/m47b/**tier5**/**m80**） |
| 該分支 `make_submission._ANCHOR` | `results_M74_default.json` = `1.293461035226291`（本機 tier 關時該有的值） |

⇒ `m80_probe` 是從這棵樹 seed 出來的 sandbox；**M80 tier 本來就從這棵樹出貨**，要移植的只有 route A。
順帶：該分支最新 commit 已把「擴池 R=512」判 **RED**（held-out 更差）⇒ 交接檔 §5 的備案 S-B 可以劃掉。

### 兩個「看起來像送件樹但不是」的東西

1. **`teammate_m71_screen` 不是送件樹。** `_m71_env()` 預設**故意反轉成 off**（`optimizer_constructive.py:449`），
   從它打包送出去跑的是 **pre-M71 = 1.3265**。樹裡那包
   `build_submission/cadc1075.tar.gz`（2026-08-09 17:18，md5 `2954f2c1`）**不可上傳**
   ——已確認 staged `op_wrapper.py:448` 的 M71 預設是 `"0"`、全檔 **0 個 M80 標記**（49 profiles，非 94）。
2. **本地 `teammate_m29_free_aspect` 是過期 checkout**（`9f72025`，落後 6 個 commit）且有未提交改動。

---

## 2. 🚨 route A 現況不能送：兩個都是靜默的

### 2a. binary 寫死、繞過解析鏈 → 送件包裡根本沒那顆檔

```python
# m80_probe/optimizer_constructive.py:1278  (_run_profile_route_a)
exe = Path(env.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive_l108.exe")))
```

不經過 `_BIN`、不經過 `_ensure_compiled()`（bundled-linux-first → 編譯鏈 → `_binary_runs()` smoke），
`_run_profile_frame` 直接 `subprocess.run` 且**沒有 exists 檢查**。送件包只有
`constructive.cpp` + `bin/constructive_linux` ⇒ 缺檔 → `FileNotFoundError`
→ 被 `_run_profile` 的 `except Exception: return None` 吞掉 → 每隻 profile 回 None → SA fallback。

**實測**（package-shaped 樹、`--test-id 99`、`ICCAD_ADAPTIVE_CORES=48`）：

| `constructive_l108.exe` | `ICCAD_ROUTE_A` | 結果 |
|---|---|---|
| **不在** | 未設（閘→開） | **`10.0000`** + `all profiles failed; python SA fallback` |
| 在 | 未設（閘→開） | `1.2773` |
| 在 / 不在 | `0` | `1.2773` |

### 2b. 更嚴重：binary 不回應 FRAMES 時 route A **無限提交、永不結束**

`_room()` 的兩個「還不知道」逃生條款——`n_frames is None`、`max_trials is None`——
對一顆**不印 FRAMES 的 binary**（任何 pre-L108 版本）**永遠為真**，`nxt` 於是無上限地爬，
一次四個**完整 pipeline** 跑（pre-L108 忽略 `ICCAD_FORCE_FRAME_IDX`，每個 frame task 都跑全套）。

**實測**：指向 pre-L108 binary、`ICCAD_ADAPTIVE_CORES=48`，**case 99 與 case 5（n=21）各跑 600s 都沒有結果**，
期間 32 個 queue slot 全滿。這不是慢，是掛住。

⚠️ 這正是 tip 現況會踩到的：`bin/constructive_linux`（541736 bytes）是 **pre-L108** 編的。
把帶 route A 的 wrapper 配上那顆 binary 送出去 = 評分機上**跑不完**。

### 2c. 為什麼本機一定看不到

**步驟 D（cores-gated 預設開）其實早就套用了**——交接檔 §3 說「剩下：預設翻成 cores-gated on」**已過期**。
實測 `m80_probe` wrapper：

| | `cores_hi` | `_route_a_default()` | `_m80_active(120)` | `pool(120)` |
|---|---|---|---|---|
| 本機預設 | 32 | **0（關）** | `frozenset()` | 13 |
| `ICCAD_ADAPTIVE_CORES=48` | 48 | **4（開）** | `{86..93}` 8 隻 | 43 |

本機 32 核 ⇒ route A / M80 tier / tier-5 **三個閘全不觸發**；
而 `make_submission.py verify` 會 `env = {k:v for ... if not k.startswith("ICCAD_")}` 把 `ICCAD_*` 剝光
⇒ **現行送件鏈的 verify 在結構上驗不到 ≥40 核組態**，它只驗證「三個 tier 全關」那一版。

---

## 3. 已建好的送件樹：`C:\ICCAD_ml\ship_final`

`git clone --branch m29-free-aspect` @ `2aae61c`，工作區乾淨後才動手。

**環境**：`LiteTensorDataTest` / `floorset_lite` 兩個 junction；`constructive.exe` 用
`C:\msys64\ucrt64\bin\g++.exe -O3 -std=c++17` 編（⚠️ msys **有裝但不在 PATH**，每次都要自己加）。

**先立基準**（動手前）：無 `ICCAD_*` 跑官方 eval →
`results_M74_default.json` = **`1.293461035226291`**、100/100 feasible、**零 fallback 行**，
且對 `results_L111_m80_48c_notier.json` **逐案 cost 100/100 + positions 100/100 相同**。

### 改了什麼（4 改 + 3 新）

| 檔案 | 內容 |
|---|---|
| `optimizer_constructive.py` | route A 移植（= m80_probe 的 2 hunk，+304/−1，git diff 逐字確認只有這兩塊）＋ §4 的四個修法 |
| `constructive.cpp` | 換成 L108（+27/−1、4 hunk；`FORCE_FRAME_IDX=-1`/`FRAME_REPORT=false` 時 `frms` 直接 alias，惰性） |
| `make_submission.py` | 併檔尾端拿掉 `import time` + 加相依 pin（route A 讓 wrapper 也 import time → AST 撞名，**實測** `AssertionError: module-level name collisions: ['time']`） |
| `bin/constructive_linux` | 用 L108 在 WSL 重編：g++ 15.2.0、`-O3 -std=c++17 -static-libstdc++ -static-libgcc`、md5 `5f65fcae`、`ldd` 只有 vdso/libm/libc/ld-linux、**會回 `FRAMES 25 5` / `FSEL 1 0`** |
| `l113_ship_gate.py` | 新：**package-shaped + 強制核數**的閘（見 §5） |
| `results_M74_default.json` | 新：make_submission 的錨（tip 上沒進 git） |
| `results_M80_48c_anchor.json` | 新：≥40 核組態的逐案錨 = `1.2666234250706565` |
| `m47b_proxy_equiv.py` / `m48_coldstart_dryrun.py` | 兩個擋路的可攜性 bug（見 §5b） |

現況 tar：`build_submission/cadc1075.tar.gz` md5 `128e434d27a407a10ff575d87e29bb32`（345192 B、6 檔），
`op_wrapper.py` md5 `3e5b71544d0b5fb6b83fc39843228999`。**未上傳，也還沒過 Linux 閘。**

---

## 4. 四個修法

1. **route A 改用 `_BIN`**（`_ensure_compiled()` 解析出來的那顆）⇒ 繼承 bundled-first + 編譯鏈 + smoke。
2. **盲目提交只限開場那一批**：`_room()` 加 `and (n_frames is not None or nxt < inflight)`。
   ⇒ 2b 的無限迴圈變成「開場 4 個 task 後就收攤」。
3. **區分兩種「route A 沒結果」**：`return {"out": None, "answered": n_frames is not None}`。
   - `answered=True`（binary 有回、只是沒有 frame 勝出）→ **回 None，維持既有行為**（48c on/off 逐位閘就是這樣驗的）。
   - `answered=False`（binary 從沒印過 FRAMES）→ 退回循序路徑，並在 stderr 印**含 `fallback` 字樣**的一行，
     讓 `make_submission.py verify` 與 `l113_ship_gate.py` 的掃描抓得到。
4. **process 級 latch**（`_ROUTE_A_DEGRADED != 0` 就不再走 route A）。
   ⚠️ 誠實範圍：`_solve_impl` 同時起所有 profile thread，**第一案所有 profile 仍會各付一次**，
   latch 省的是第 2..N 案。

> 我第一版的降級訊息把原因寫死成「binary 不支援」，結果在正常 L108 binary 上也印了——
> 那其實是「這隻 profile 沒有 frame 勝出」的良性情況。已按上面第 3 點改掉。

---

## 5. 驗證結果

| 閘 | 判準 | 結果 |
|---|---|---|
| **G1 移植後不動既有行為** | 32 核預設 vs 動手前的錨 | **逐位相同**：total `1.293461035226291`、cost 100/100、positions 100/100 |
| **G2 48 核 route A on vs off** | 逐位 + peak ≤ cores | **逐位相同**：total `1.2666234250706565`、cost 100/100、positions 100/100、peak `32 32` |
| **G2b 跨樹** | ship_final 48c off vs `m80_probe/results_L111_m80_48c.json` | **逐位相同** 100/100 |
| **G3 負向：pre-L108 binary** | 必須吵、且不可掉分 | 印出 `route A binary answers no ICCAD_FRAME_REPORT (pre-L108?); sequential fallback`，分數 `1.4299` **= route A 關掉的對照值** |
| **G4 封包級 48 核閘** | `l113_ship_gate.py --cores 48` | **ALL PASS**：解開的 tar 用官方指令跑 → total `1.2666234250706565`、feasible 100/100、cost 100/100、positions 100/100、零 fallback、peak `32/32`、包內自行編出 binary（310s） |
| **G5 Linux 100 案逐位** | `verify_final_tar.sh` | ⏳ **未做——本機 WSL 沒有 numpy/torch/shapely** |

G4 是這輪最有價值的一道：它跑的是**真的送件包**（op_wrapper.py 併檔形、包內編譯、官方目錄佈局），
而且**強制 48 核**把三個 tier 都打開——現行 `make_submission.py verify` 這兩件事一件都沒做到。
它同時順帶證明併檔形與 repo 形 cost-identical（= `m48_coldstart_dryrun.py opwrapper` 的判準）。

---

## 5b. `regression_suite.py`：能跑的 4 項全綠，3 項卡在快取

| 項目 | 結果 |
|---|---|
| `m48` coldstart 四 phase | **ALL PASS**（85s；phase 1 冷編 L108 源碼、phase 2 垃圾 binary 被 smoke 攔下重編、phase 3 bogus 編譯器被跳過） |
| `m47b` proxy 等價 | **PASS**：300 comparisons、mismatches **0** |
| `tier5` 池身分 | **ALL PASS**（≤39 核惰性、≥40 核 n>100 恰等於 restore-knob、fail-closed） |
| `m80` tier 身分 | **ALL PASS**（含 `V3 shipped prefix == HEAD (41 profiles)` = 四份離線快取仍有效） |
| `rf` / `m49big` / `m49mid` | ⏳ **跑不了**（見下） |

**順手修掉兩個擋路的可攜性 bug**（都是 offline 工具、永不出貨，跟 route A 無關）：

- `m47b_proxy_equiv.py:13` `REPO` **寫死成隊友機器的 checkout 路徑** ⇒ 判定「資料集不存在」
  → `lite_dataset_test` 去**下載**到同一個不存在的路徑 → `FileNotFoundError`。改成 `Path(__file__).resolve().parent`。
- `m48_coldstart_dryrun.py:47` `subprocess.run(text=True)` 用**locale codec** 解碼子行程 stderr；
  這台是 zh-TW ⇒ cp950 解不了編譯器輸出 → reader thread 死 → `err3` 變 `None` →
  phase 3 `TypeError`。改成明示 `encoding="utf-8", errors="replace"`。

## 6. 還沒做的（照順序）

1. **`rf` / `m49big` / `m49mid` 三個閘需要 `audit_cache.pkl` + `m49_pm_cache.pkl`**，
   兩者都被 `.gitignore` 擋掉、**fresh clone 沒有**；screen 樹那三份是 49-profile 池的、**signature 不相容**。
   要嘛跟隊友要，要嘛用 `profile_audit.py` 重生（94 profiles × 100 案，很貴）。
   ⚠️ 但這幾個閘管的是「池常數推導」，而我們**沒動 `_PROFILES`**（`m80` 閘的 V3 也剛證實
   shipped prefix 與 HEAD 相同）；動到的 `constructive.cpp` 已由 G1/G2/G4 三組 100 案逐位相同
   直接證明沒翻任何 proxy tie——比常數閘更強的證據。
2. **Linux 端 100 案逐位驗證（必做，且是唯一沒被覆蓋的風險）**：
   `bin/constructive_linux` 這次是 **g++ 15.2.0** 編的，先前 M67-C 那顆是 Ubuntu 22.04 / **g++ 11.4.0**。
   來源與編譯器都換了 ⇒ **上傳前一定要在 Linux 機跑 `verify_final_tar.sh`**。
   若手上有那台 22.04 機器，建議在那邊重編以保持 provenance。
3. LP（L112）要不要上：照交接檔 §5.2，驗證必須是官方 eval + OOS。

---

## 7. 寫報告時務必分清楚的一件事

- **route A = 真逐位相同**：FSEL 在後處理前印出 ⇒ 選擇鍵不變；本輪 G1/G2/G2b 三組 100 案
  逐案 cost + positions 全等。等價閘足以當最終驗證。
- **形狀 LP ≠ 逐位相同**：只保證**目標值**相同，佈局非逐位（LP 高度退化，dq ~2.6e-12）。
  **最終驗證必須走官方 eval + OOS**，不能只靠等價閘。

兩者不可混為一談。
