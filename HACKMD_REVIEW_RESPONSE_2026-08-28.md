# 回覆 HackMD 驗收意見 A–H — 2026-08-28

對象：`https://hackmd.io/N-dy6PvzRc6qsmf4Z0uV_A` 的 §三「沒提到、我認為該做的工作」A–H。
每一條的判定 + 證據 + 改了什麼。**四條 `verify_kit/*` 連結是相對路徑、指向對方未 commit
的樹，抓不到**；其中 `l246_compliance.py` 是本樹的檔原封搬過去的，已直接用本樹版本複驗。

| | 主題 | 判定 |
|---|---|---|
| A | deadline 對不上 | ✅ **08-31 23:59 GMT+8**，文件已修 |
| B | median 漂移敏感度 | ✅ 表 + 決策規則做出來了，**前提方向與原假設相反** |
| C | 資料不在 git / 硬編路徑 | ✅ 兩處硬編修掉，bundle 已備妥 |
| D | 絕對路徑 patch | ✅ 已套用並驗證，**但 patch 的理由有一句是錯的** |
| E | Case B venv 沒端到端跑過 | ✅ 解析跑了，**抓到一個新風險：scipy 1.18.1 ≠ 1.18.0** |
| F | Drive 衛生 | ✅ 本機部分修了，**七顆同名 tar 只有一顆有護欄** |
| G | 驗收流程 | ✅ 清單做出來了，**但 D 無法從自己的 commit 重建** |
| H | 對方跑不了 Linux 車道 | ✅ 三個宣稱全部複驗過 |

---

## A — deadline：**2026-08-31 23:59 GMT+8**（Problem C 專屬）

兩份主辦文件都對，因為**延期是按題目分的**：

* `Final Submission Guidelines_ABC.pdf`（`~/Downloads`，08-25 18:52 下載）：
  "postponed to August 28, 2026 (17:00, GMT+8)" — **ABC 通用**。
* 主辦澄清信（使用者本次提供）：延到 08-31 23:59 **只適用 Problem C**；其他題維持 08-28 17:00。

我們是 Problem C（`README.md:5`、team `cadc1075`、走 B/C 的 Google Drive 路線）⇒ **08-31**。
判定當下是 08-28 12:24，剩 **3 天 11.5 小時**，不是 4.6 小時。

**🔑 可外推的教訓**：`ledger` 已經有「primary text is the authority」，這次要補一句——
**主文件也可能被一份「從沒提到你」的後續文件在你的範圍上取代**。兩份衝突時，先確認
它們**量詞的範圍是否相同**，再判誰過期。只拿到 PDF 的 session 推出 08-28 是正確推理。

已改：`CLAUDE.md` 三處（23 / 102 / 231），新增 [DEADLINE_PRIMARY_EVIDENCE.md](DEADLINE_PRIMARY_EVIDENCE.md)
（含信件原文）。歷史 handoff 保留原樣（那是當時的紀錄）。

---

## B — median 漂移敏感度表 + 決策規則

工具：[l309_median_sensitivity.py](l309_median_sensitivity.py)。**三道驗證全過**：

```
2026-08-16  got 0.924518585921  published 0.924518366998  rel 2.4e-07  PASS
2026-08-23  got 0.926586662561  published 0.926586116132  rel 5.9e-07  PASS
discrete: cases off floor on 08-16 medians = [66]  (expect [66], n=87)  PASS
```

⚠️ **誠實界定**：對照臂的 bridge 塌成恆等式，所以這三道驗的是**計分管線**（權重、RF 公式、
median 向量、兩個公布總分），**不是 bridge**。`l146` 自己的 docstring 就警告過這個形狀。
bridge 靠的是 `l285_betacfg.json` 代表 beta 包，而它是 **M73-like**（帶著 L131/L136 修正）
⇒ **絕對值偏保守**（本檔把 D 投到 ~0.900/rank 4，`HANDOFF_2026-08-29` 投 0.87511/rank 2）。
**要看的是 NET-vs-D 那一欄**，它對這個偏差免疫（所有臂共用同一分母）。

### 🔑 兩個結構性發現

1. **m 與機速 f 是共線的**：`R_i = t_i^local / (f · m · M_i)` ⇒ 只有乘積有意義。
   一張表同時是 median 敏感度表與機速敏感度表。L308 的 f = 2.38–2.84 本身就是 1.19× 的帶。
2. 🚨 **唯一觀測得到的那次 median 漂移是「變小 26%」，不是變大**：
   08-16 → 08-23 逐案 `M23/M16` p10 0.600 / **p50 0.742** / p90 0.876，離散度 1.95×。
   原意見的前提（頂端隊伍都跑 110s ⇒ Final median 變大 ⇒ 那 2.69% 品質白付）
   **與唯一的實測相反**。可能仍會發生，但不是 base case。

### 表（medians = 08-23，乘上 m；cell = 總分 / 名次 / 離開 RF floor 的案數）

```
     m |          D (shipped) |  REFINE restored 4/6 |    LP gate 100       |    LP off (floor ref)
 0.742 |     0.91284 r4  35off |     0.96081 r5  55off |     0.93655 r5  46off |     0.93700 r5  20off
 0.800 |     0.90602 r4  16off |     0.94242 r5  40off |     0.92307 r4  36off |     0.93160 r5  10off
 0.900 |     0.90098 r4   3off |     0.92049 r4  21off |     0.90774 r4  22off |     0.92631 r4   5off
 1.000 |     0.90005 r4   0off |     0.90768 r4   9off |     0.89735 r3  15off |     0.92411 r4   2off
 1.100 |     0.90005 r4   0off |     0.90297 r4   2off |     0.88977 r3   8off |     0.92409 r4   0off
 1.200 |     0.90005 r4   0off |     0.90169 r4   1off |     0.88509 r2   4off |     0.92409 r4   0off
 1.300 |     0.90005 r4   0off |     0.90082 r4   1off |     0.88116 r2   4off |     0.92409 r4   0off
```

NET vs D（正 = 比出貨 D 好）與交叉點：

```
 m      REFINE restored   LP gate 100
 0.742      -5.255%          -2.597%
 1.000      -0.848%          +0.299%
 1.300      -0.086%          +2.098%

 REFINE restored  在 m ∈ [0.2, 6.0] 全區間都追不上 D
 LP gate 100      m >= 0.97 才追得上
 曝險（m ∈ [0.742,1.3] 的擺幅）：D 0.0128 / REFINE 0.0600 / LP100 0.0554
   -> D 對 median 漂移的曝險小 4.7 倍
```

### 決策規則（事前註冊）

**m 在截止前不可觀測**（Final median 是跨隊的、評分後才公布）⇒ 不能「等著看」，只能押先驗。

> **RULE：出 D。**
> * **REFINE 還原**：任何 m 都不會贏 ⇒ 無條件關閉。
> * **LP 閘開滿 100**：只有 m ≥ 0.97 才贏，也就是要賭 Final median **至少和 08-23 一樣寬鬆**
>   ——賭的方向與唯一觀測到的那次相反。上檔最多 +2.1%，下檔（若照上次那樣漂）−2.6%。
> * D 同時是最平的臂（曝險小 4.7 倍）。**它的名次不依賴一個我們看不見的數字**，
>   這件事本身比它的點估計值錢。
> * 會改變判定的只有兩種東西：拿到 Final 的 median 向量，或一個**等牆鐘、純品質**的機制。
>   只要 m 未知，任何「拿時間換品質」都過不了這道門。

---

## C — 硬編路徑與資料交付

* `l146_rf_price.py:55` 的 `C:/Users/.01/Downloads/...` → 換成 `_median_csv()`：
  repo 內 → `$ICCAD_MEDIAN_CSV` → 舊的 Downloads 位置。
* `l276_price.py:35` 同一條，改成沿用 `l146` 的解析。
* **回歸驗過，數字一個都沒動**：`l146 curve` 仍是 `0.9245185859`（官方 `0.9245183669982832`，
  rel 2.4e-7），`l276` 仍是 `0.9265866626`。
* median CSV 已複製進 repo：`C_median_runtimes_beta_hidden.csv`。

**交付包**：`handover/rf-pricing-data-2026-08-28.tar.gz`
（1,730,880 B、19 個成員、md5 `17ae6ef7a7bb39a273857300236250cf`）。
**已從乾淨解壓驗過自足**：在沒有 `C:/Users/.01/Downloads` 的目錄下 `l309` 與 `l146` 都跑得出
同樣的數字。內含 beta 08-16 逐案結果 + 整個 08-23 + 兩份 median + 三支 pricing 腳本 +
l309 需要的六個 arm results。

進 git（**我沒有 commit，這是你的決定**）：

```bash
git add -f beta_2026-08-16/beta_evaluation_results.json beta_2026-08-16/eval_op_wrapper.log beta_2026-08-23/ C_median_runtimes_beta_hidden.csv
```

---

## D — 絕對路徑 patch：已套用，但更正 patch 說明裡的一句話

套用結果（`git apply`，兩檔都自 `8f2abc4` 未動 ⇒ 乾淨套用）：

* 用 `make_submission` **自己的** `_ABS_RE` 物件掃（**不可重打**，見下）：
  D 現況 **2 hit**（`op_wrapper.py:1709` + `op_src.py:1709`，與對方獨立掃出的完全一致）
  → 套用後 **0 hit**，其餘四個文字檔本來就是 0 ⇒ **拿掉 `_ABS_ALLOW` 不會誤傷任何東西**。
* `l246_compliance.py` 對 D 是 **19/20**，唯一 FAIL 就是 `no absolute paths in code`；
  對套用後的包，該條 **OK**。

### 🚨🚨 這一節我先前寫錯了，已更正（2026-08-28 晚）

**先前的版本說**：patch 註解裡「裸 `g++` 解到同一顆 msys binary」是錯的，因為
`shutil.which()` 對三個編譯器都回 None，而那條絕對路徑「當下是能跑的
（`returncode 0`）」。

**那個 `returncode 0` 是 `--version`，而 `--version` 從來不會叫起 `cc1plus`。**
用真正的編譯重測：

```
A. --version（只跑 driver）
   絕對路徑、PATH 不動        rc=0        <- 我先前唯一的證據，是錯的探針
B. 真的編譯（會叫起 cc1plus）
   絕對路徑、PATH 不動        rc=1        產物 NOT CREATED
   絕對路徑、msys 在 PATH 上  rc=0        產物 255,002 bytes
C. 裸 g++
   which(g++) 預設 PATH       None
   which(g++) msys 在 PATH    C:\msys64\ucrt64\bin\g++.EXE   <- 同一顆
```

⇒ **機制**：`cc1plus.exe` 住在 `lib/gcc/…`，它的 DLL 在 `ucrt64/bin`。用絕對路徑叫
`g++` 時，driver 找得到自己的 DLL（同目錄），但 **`cc1plus` 找不到自己的**，於是無聲死掉、
driver 回 1 且沒有任何診斷訊息。

**兩個更正**：

1. **patch 的註解是對的**——只要 `ucrt64\bin` 在 PATH 上，裸 `g++` 就是解到同一顆。
   我先前判它為假，判錯了。真正成立的是更窄的一句：**沒有 PATH 時兩種寫法都不能編譯**，
   因為壞的是 `cc1plus` 不是 driver ⇒ **那條絕對路徑本來也沒買到任何東西**。
2. 🚨 **`ICCAD_CXX` 修不好這件事**（我先前的建議是錯的）。把 `ICCAD_CXX` 指到那條絕對
   路徑會撞上**同一個** `cc1plus` 失敗。**正解是 PATH：**

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"      # PowerShell: $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
```

移除在**評分機上成本仍然恰好為零**（POSIX 走 bundled ELF，`os.name != "nt"`）。
`l113_ship_gate._cxx_preflight` 已經在做 PATH prepend，所以那條車道一直是安全的。

⚠️ **在 Windows 上驗證的人沒設 PATH 會拿到 `Total Score: 10.0000 / Feasible: 100/100`**
——那是 SA fallback 的 *feasible 上限* 9.999999 被 `%.4f` 進位，**不是全案不可行**。
這正是 ledger 的 `windows-msys-path-silent-sa-fallback`。

### 🔑 方法論：量測工具不可以「重打」被測對象

我第一版把 `_ABS_RE` 逐字重打進腳本，shell 把 `\\` 併成 `\`，字元集從「反斜線或斜線」
變成「只有斜線」，於是**對 D 掃出 0 hit** —— 正好漏掉唯一要抓的那一行。
自檢（已知答案的正反例）當場擋下來。對方也踩過同一個坑（lookbehind 排掉引號）。
**規則：直接 import 被測模組拿它的物件，不要重打 pattern。**

---

## E — Case B 乾淨 venv：解析跑通了，但抓到一個新風險

`pip install --dry-run -r requirements.txt`（Python 3.13）**解得乾乾淨淨、零衝突**，
主要解到：

```
torch-2.13.0  numpy-2.5.2  shapely-2.1.2  scipy-1.18.1  matplotlib-3.11.1
```

⇒ §4(a) 記載的 Alpha 失敗模式（「pip 解到不相容版本」）**在這組 `>=` 下限上不會重演**。

### 🚨 但是：runbook 的期望值是在 **scipy 1.18.0** 上驗的，pip 現在解到 **1.18.1**

`1.2264069637381392` 這個數字的成立條件包含 scipy 版本——shape LP 高度退化，
Windows 1.15.3 與 Linux 1.18.0 就已經落在**同一個 LP 的不同最佳解**（8/100 案不同，L119）。
**minor 版本跳動屬於同一類擾動**，沒有理由假設 1.18.1 一定落在同一個頂點。

處置（建議**不要**動）：

* **不要 pin**。`l246` 有一條 `no pinned versions (>= only), so Python 3.13 can resolve`，
  pin 會把一個「數字可能小動」的風險換成一條**確定的 compliance FAIL**。
* 下檔是有界且已知的：runbook §3 已記載 LP 整條關掉是 `1.2589744529416786`、**仍 100/100 feasible**、
  不會炸。scipy 換頂點的影響遠小於這個界。
* **驗收時把期望值當區間讀**：`total` 落在 `1.2264` 與 `1.2590` 之間、feasible 100/100、
  `lp_stats.txt` 71 行 ⇒ 通過。只有 feasible 掉了或 `total ≈ 10` 才是真迴歸。

---

## F — Drive 衛生

### 🚨 本機發現：七顆同名 `cadc1075.tar.gz`，原本只有一顆有護欄

```
build_submission              35,513,422 B  1544 members  vendor  op_wrapper 1c326784…  (有護欄)
build_submission.D               408,795 B     8 members  ——      op_wrapper 1c326784…  ← 要傳的
build_submission.NOVENDOR        408,794 B     8 members  ——      op_wrapper 1c326784…
build_submission.L131FIX         376,840 B     8 members  ——      op_wrapper 2967efb6…  ← 舊包
build_submission.L136FIX         377,124 B     8 members  ——      op_wrapper 2967efb6…  ← 舊包
build_submission.MIX             621,942 B     8 members  ——      op_wrapper b74cadae…  ← 舊包
build_submission.SHIPPED.bak     373,844 B     8 members  ——      op_wrapper ad8c5dcb…  ← 舊包
```

`L131FIX` / `L136FIX` / `MIX` / `SHIPPED.bak` 是**更舊、更差**的包，檔名一模一樣，
而**護欄只有 `build_submission/` 有**。這正是 ledger 記過一次的「人為抓錯目錄」。
已對六個非 D 目錄各寫入 `DO_NOT_UPLOAD.txt`（含該目錄自己的 op_wrapper md5 與正確目標）。

順帶把 `.gitattributes` 的 CRLF 防護一般化：原本是**一次一條路徑**加了五次，
`MIX` / `SHIPPED.bak` 從來沒被涵蓋，未來的 `build_submission.E` 也不會。已加

```
build_submission*/** -text
```

實測 `git check-attr` 對七個現有目錄 **與假想的 `build_submission.E` 全部 `text: unset`** ⇒ 陷阱拆掉了。

### 我做不到的（需要人去 Drive 上確認）

1. 資料夾名是 **`final_test_submission`**、在 **cadc1075** 自己的 home，不是 Beta 資料夾、
   不是別隊的 `cadb1036`（曾傳錯一次）。
2. **該資料夾裡只有一顆 tar**。舊的 vendor 版或舊的 D 躺在旁邊，評分機挑哪顆是未定義的。
3. 檔名必須恰好 `cadc1075.tar.gz`——主辦白紙黑字寫**檔名錯誤不接受重傳**
   （Chrome 重新下載會加 ` (n)`，別把那個名字傳上去）。

---

## G — 驗收流程

### 🚨 頭號發現：**D 無法從它自己的 commit 重建**，所以 md5 硬比對會給無意義的 FAIL

`build_op_wrapper_text()` 是決定性轉換。用 **`8f2abc4` 的原始檔**（也就是
`final-2026-08-26-verified` 的 tip，D 應該來自這裡）跑出來是 `fbf55d41…`，
**不是 D 出貨的 `1c326784…`**。逐行 diff 只有**兩個 hunk，全都是 L240 的
`ICCAD_LP_EDGE_WEIGHT` 探針**，預設 OFF（未設環境變數時傳給 scipy 的參數完全相同）。

⇒ D 是在 L240 探針進樹**之前**打的包。三個 md5：

| | staged `op_wrapper.py` md5 |
|---|---|
| **D（已上傳）** | `1c326784de7cd9246cd1f380e2842668` |
| 現行樹，未套 patch | `fbf55d4138afd62bb78cca68c95e3998` |
| 現行樹，**套了 patch** | **`481ee68009410050496fa1bc4fb02bac`** |

**所以驗收閘要改**：不能寫「op_wrapper md5 必須等於 `1c326784…`」——星期日那顆一定不等於，
**就算不套絕對路徑 patch 也不等於**。正確的閘是下面清單的 ②+④。

### ✅ 已跑完整 stage + l246：**20/20**

第一次跑 `stage()` 時 FAIL（`vendor/ carries 262 .pyc (a local import polluted it)`），
重跑時 `vendor/` 已經乾淨（1418 檔、0 `.pyc`、sha256 與官方 wheel 相符、extra/missing 皆空）。

### 🚨 歸因找到了：**另一個 session 正在同一棵樹上工作**

`build_submission/` 在 12:57–12:58 被重新 stage、`DO_NOT_UPLOAD.txt` 在 13:21 被改寫，
兩者都不是我做的。對方留下的字條寫得很清楚：

> This stage currently holds the **L303 MIX CANDIDATE**, not the shipped package:
> `op_wrapper.py md5 2b795995…`，`_L196_LPGATE` all 1s、`_L157_DEPTH` 2 on the old 1-set.
> The tree's `optimizer_constructive.py` was RESTORED after staging
> (**a second session is editing it**)

⇒ 那 262 個 `.pyc` 是對方 import 過 vendored scipy 留下、之後清掉的。
⇒ **`build_submission/` 現在裝的是 L296-L298 的 `mix` 臂**（LP 閘開滿 + 深度 k=2），
不是出貨包；他們另外放了 `build_submission.MIXD/`（同 md5 `2b795995`）。
兩邊都偵測到對方、都留了字條，**`build_submission.D/` 兩邊都沒動過**（我最後複驗：
`1c326784…`、408,795 B）。

⚠️ **但這對驗收有直接影響**：這棵樹現在有兩個寫入者，**任何在這裡做的量測都可能被競態**。
我的數字是在特定時刻取的，收尾時全部重驗過一次（見下）。
按 B 節的決策規則，`mix` 正是「LP 閘開滿」那一族——**在 m < 0.97 時輸給 D**。

處置：重 stage 前加一道硬檢查，並確認沒有第二個 session 在跑：

```bash
python -c "import pathlib,sys; n=sum(1 for _ in pathlib.Path('vendor').rglob('*.pyc')); print(n); sys.exit(n>0)"
```

完整結果（stage 導到 scratch，**沒有動 `build_submission/`**）：

```
stage()  -> True   vendor OK (1418 files, byte-identical to the official wheel)
                   hygiene OK (1424 files)
l246 on the VENDOR tar      -> 多條 FAIL（vendor/ 本來就過不了 A 62-68，這正是 L245 反轉的原因）
l246 on the NO-VENDOR tar   -> **20 / 20 checks pass**    409,678 B, 8 members
```

無 vendor 版的成員清單與 D **逐項相同**（8 個）。

### ④ 逐檔 diff vs D — 恰好是預測的三處，其餘逐位相同

```
op_wrapper.py        D 1c326784…  NEW 481ee680…  DIFFERS  3 hunks  4729 -> 4746 行
op_src.py            D 1c326784…  NEW 481ee680…  DIFFERS  3 hunks  (op_wrapper 的副本)
constructive.cpp     e2c7b2f418ef2b70b6bff99f7adfbd37   IDENTICAL
README.md            8167121dbb2a0bcb05851226a6f6a4e3   IDENTICAL
requirements.txt     6c59feb458f1f48247373d8b69f401c2   IDENTICAL
bin/constructive_linux  bc9912072cd97b45b47a03adec7170ce  IDENTICAL
```

三個 hunk：①註解改寫（描述而非引用那條路徑）+ `if os.name == "nt"` 兩行移除（同一個 hunk）、
②L240 `_ew = os.environ.get("ICCAD_LP_EDGE_WEIGHT", "")`、③L240 的 `linprog` 呼叫。
②③**不是這個 patch 帶來的**，是 D 打包之後才進樹的探針，預設 OFF。

⇒ **`bin/constructive_linux` 與 `constructive.cpp` 都逐位不變，確認不需要 Linux 重建。**

**收尾複驗（在發現有第二個寫入者之後全部重跑一次）**：

```
current source -> staged op_wrapper md5   481ee68009410050496fa1bc4fb02bac   (不變)
_L157_DEPTH   100 entries, 全是 1                                            (D 的組態，非 mix)
_L196_LPGATE  100 entries, 71 開 / 29 關                                     (D 的組態，非 mix)
l246 on my no-vendor tar                  20 / 20 checks pass                (不變)
build_submission.D/cadc1075/op_wrapper.py 1c326784de7cd9246cd1f380e2842668   (未被動過)
results_L237_post.json / l285_betacfg.json                                   (未被動過)
```

### ⚠️ patch 的註解把一句已證實為假的話寫進了出貨檔

新的 `op_wrapper.py:1712` 註解裡有

> "locally it is redundant -- a bare g++ resolves to the same msys binary"

**這句在這台機器上是假的**（見 D 節：`g++`/`clang++`/`c++` 三個都不在 PATH）。
它只是註解、不影響分數，但會誤導未來的維護者把 `ICCAD_CXX` 那條退路也拿掉。
**我沒有自作主張改它**——那是對方 patch 的原文，改了對方就沒法用 `git apply` 複驗。
要改的話一行就好，改完 md5 會再變一次。

### 交件清單（六項，對方原本要的 + 兩項修正）

| # | 項目 | 值 / 怎麼給 |
|---|---|---|
| ① | 期望的 48c Linux total | `1.2264069637381392`，feasible 100/100。**當區間讀**：`[1.2264, 1.2590]` 都算過（見 E 的 scipy 1.18.1） |
| ② | `op_wrapper.py` md5 | 新包實際的值（**不是** `1c326784…`）。套 patch 後預期 `481ee680…`，但要以實際 stage 產物為準 |
| ③ | `lp_stats.txt` 行數 | **71** |
| ④ | **vs D 的逐檔 diff** | 必須逐 hunk 說明。目前預期**恰好三處**：L240 探針（預設 OFF）、msys 那行、其上方註解。**出現第四處就停下來** |
| ⑤ | 五車道 log | `l238_wsl_final.sh`，`L238_WSL_RC=0` |
| ⑥ | kill switch 名稱 | `ICCAD_L223_REFINE_HEAVY`、`ICCAD_L231_REFINE_MID`、`ICCAD_LP_GATE`、`ICCAD_SHAPE_LP`、`ICCAD_M80_TIER`、`ICCAD_M67F_TIER5` |

### 事前註冊的驗收門檻（**在看到新數字之前訂死**）

1. **0-regression vs D**：對 `results_L237_post.json` 逐案比，**沒有任何一案 cost 比 D 差**。
   有任何一案變差 ⇒ 必須定價，不能用「總分還是比較好」帶過。
2. **positions 100/100 逐位相同**（絕對路徑 patch 不碰任何求解路徑；**動了就是套錯地方**）。
3. **feasible 100/100**。低於這個直接停。
4. `wc -l lp_stats.txt` = 71。0 行 = scipy 沒裝到；100 行 = `ICCAD_LP_GATE=0` 殘留在環境。
5. log 裡 `SA fallback` **0 次**，`[scipy] source=` 有值。
6. **`bin/constructive_linux` md5 若變了**：必須附重建證據 + `strings` 複驗（`ICCAD_*` 逐個
   出現）。**沒變**則要求 `constructive.cpp` 也逐位沒變（本次 cpp 自 08-19 未動 ⇒ **不需要
   Linux 重建**）。
7. `l246_compliance.py` **20/20**。
8. `l113_ship_gate.py --cores 48` 全綠 —— **這條只有我們這台跑得動**（對方每個編譯器 exit 1）。

### 凍結時點

**08-30 23:59 GMT+8**（截止前 24 小時）。過了就守 D。
D 已上傳且驗過，是已知良品；用最後一天去換一個沒驗完的包，期望值是負的。

---

## H — 對方的 Linux 車道

三個宣稱都複驗過：

1. ✅ **核數不影響分數**。官方 harness 自己印
   `NOTE: Local runtime scores use RuntimeFactor=1.0 (neutral).`
   （`iccad2026_evaluate.py` ~925-940）⇒ RF 在本地恆為 1.0。
   `ICCAD_ADAPTIVE_CORES=48` 把池形狀釘死 ⇒ **16 核也能算出同一個數字，只是慢**。
2. ✅ **scipy 版本會改變數字**。本機 base 是 **1.16.3**、`floorset` env 是 **1.15.3**，
   兩個都不是 1.18.x ⇒ 用它們跑會落在 Windows 那個頂點（`1.2263251265`，8/100 案不同）。
   要逐位重現 `1.2264069637381392` 需要 Linux + scipy 1.18.x。
   ⚠️ 但見 E：pip 現在解到 **1.18.1**，連 1.18.0 都不保證了 ⇒ **改用不變式判，不要用逐位**。
3. ⚠️ **WSL 沒裝 distro**。裝法：

```bash
wsl --install -d Ubuntu-24.04
```

裝完在 distro 內：`sudo apt install -y g++ python3-venv`，然後照
`VERIFY_RUNBOOK_2026-08-27.md` §2 佈置目錄、建 venv、跑官方指令。
`constructive.cpp` 自 08-19 未動 ⇒ **不需要重建 ELF**，`chmod +x bin/constructive_linux` 即可
（它需要 GLIBC ≥ 2.38；Ubuntu 24.04 是 2.39，過關）。

---

## 我改了什麼（檔案層級）

| 檔案 | 動作 |
|---|---|
| `optimizer_constructive.py` | 套用 patch：拿掉 nt-gated msys 絕對路徑 |
| `make_submission.py` | 套用 patch：拿掉 `_ABS_ALLOW` 白名單 |
| `l146_rf_price.py` | `_median_csv()` 取代硬編 Downloads 路徑（+ `import os`） |
| `l276_price.py` | 同上，沿用 l146 的解析 |
| `l309_median_sensitivity.py` | **新增** — B 的敏感度表 + 決策規則 |
| `DEADLINE_PRIMARY_EVIDENCE.md` | **新增** — A 的證據鏈 |
| `CLAUDE.md` | deadline 三處 08-28 → 08-31 |
| `.gitattributes` | 加 `build_submission*/** -text` 通配 |
| `build_submission.{L131FIX,L136FIX,MIX,NOVENDOR,SHIPPED.bak}/DO_NOT_UPLOAD.txt` | **新增** 五份護欄 |
| `C_median_runtimes_beta_hidden.csv` | 從 Downloads 複製進 repo |
| `handover/rf-pricing-data-2026-08-28.tar.gz` | **新增** — 給對方的資料包 |

**沒有 commit、沒有重 stage、沒有動任何求解路徑、沒有碰 `build_submission.D/`。**

## 還沒做的（需要你決定或需要人）

1. **Drive 上的三件事**（F 節末）——資料夾名 / 只有一顆 tar / 檔名逐字。
2. **是否 commit** 資料檔與這些修改。
3. **要不要改掉 patch 註解裡那句假話**（G 節末）。改了 md5 會從 `481ee680…` 再變一次，
   而且對方就沒法用 `git apply` 逐位複驗他們自己的 patch。
4. **要不要把新包寫進 `build_submission.E/`**——目前只在 scratch。`.gitattributes` 的
   通配已經涵蓋那條路徑了。⚠️ 但按 B 的決策規則，**現在的建議是出 D**，這顆新包的用途是
   「D + 一條 compliance 修補」，不是換演算法。
5. **Linux 五車道**（`l238_wsl_final.sh`）——`constructive.cpp` 與 ELF 都逐位不變，
   照理不需要，但按 G 的門檻 ⑤ 仍要跑一次拿 log。
