# RF-SAFE 出貨包 — 驗收文件（2026-08-28 18:30 GMT+8）

給做同儕審查的人。**你先前那份報告是對 `build_submission.D` 寫的**（`op_wrapper.py`
md5 `1c326784…`、LP 71 blocks、48c Linux `1.2264069637381392`）。這份包是**不同的一顆**，
所以下面每一道閘都是**對 RF-SAFE 重跑的**，不是沿用 D 的結果。

---

## 一、要上傳的檔案

```
C:\ICCAD_ml\ship_final\build_submission.RFSAFE\cadc1075.tar.gz
```

| | |
|---|---|
| 大小 | **409,589 bytes** |
| tar 成員 | **8**（無 `vendor/`） |
| **身分：`op_wrapper.py` md5** | **`62db6ee4569b31ddc8c546ccf3e7cd0b`** |

🚨 **不要用 tar 的 md5 當身分** —— gzip 內嵌 mtime，每次 stage 都會變。**一律看
`op_wrapper.py` 的 md5。**

逐檔 md5（2026-08-28 18:28 現場重驗）：

```
op_wrapper.py            62db6ee4569b31ddc8c546ccf3e7cd0b     DIFFERS from D
op_src.py                62db6ee4569b31ddc8c546ccf3e7cd0b     DIFFERS from D
constructive.cpp         e2c7b2f418ef2b70b6bff99f7adfbd37     same as D
bin/constructive_linux   bc9912072cd97b45b47a03adec7170ce     same as D
README.md                8167121dbb2a0bcb05851226a6f6a4e3     same as D
requirements.txt         6c59feb458f1f48247373d8b69f401c2     same as D
```

**只有 wrapper 變了。`constructive.cpp` 與 ELF 逐位等於 D ⇒ 沒有重建 ELF，
glibc floor 的問題結構上不存在。**

---

## 二、對 D 到底改了什麼（實際 diff，不是描述）

`diff -U1` = **4 hunks**（你報告裡的數字；`-U3` 會合併成 3、`-U0` 會拆成 7，
**hunk 數取決於 context 寬度，語意變更是三項**）：

1. **拿掉 `op_wrapper.py:1709` 的絕對路徑** —— 就是你 §D 標為 blocker 的那一行
   （`C:\msys64\ucrt64\bin\g++.exe`，原本在 `os.name == "nt"` 之下）。**這是 RF-SAFE
   相對 D 的淨改善**：它在評分機上永遠不可能執行（那裡 `os.name` 不是 `"nt"`），
   本機則是冗餘的（裸 `g++` 會解到同一顆 msys binary）。
2. **L240 探針** —— `ICCAD_LP_EDGE_WEIGHT`，**預設 OFF**，不設環境變數時
   `linprog(...)` 的呼叫**逐位不變**（`**({} if not _ew else ...)`）。
3. **`_L196_LPGATE` 71 → 83。**

第 3 項我做了程式化驗證（不是讀 diff）：

```
keys                 : 100  (block counts 21..120)
LP ON:  D = 71   RFSAFE = 83
newly ON   (12)      : [38, 40, 56, 76, 79, 81, 94, 95, 107, 108, 114, 120]
turned OFF ( 0)      : []          <- 沒有任何一個被關掉
```

**恰好 12 個開啟、0 個關閉。** 挑選準則是「**增加的 grader 時間塞得進該案自己到
RF floor 的 slack**」—— **完全不看品質**，所以沒有東西被擬合。

---

## 三、你列的 7 道閘，逐道對 RF-SAFE 的結果

| # | 閘 | 結果 | 證據 |
|---|---|---|---|
| 1 | **identity / 絕對路徑**（D 有 1 個 FAIL） | ✅ **0 個** | `l246` 的 "no absolute paths in code" 回空清單；D 的那一行已移除 |
| 2 | **score 獨立重算** | ✅ `1.2178289924684162`，feasible **100/100** | L313 lane 4（48c Linux，`judge48()`） |
| 3 | **determinism / diff 100/100** | ✅ | 預設核數下對 D **逐位相同含 positions**；L341 再獨立重現一次（worst \|dcost\| = 0） |
| 4 | **logcheck `scipy: source=system`** | ✅ | L313 五車道；L341 也印出 `scipy: system` |
| 5 | **`l246_compliance.py` 20 條** | ✅ **20/20** | 2026-08-28 重跑，逐條對官方文件 A/B |
| 6 | **0-regression vs D** | ✅ | 雙平台各 **12 movers / 0 worse / 0 stray** |
| 7 | **ELF 未變 ⇒ cpp 須逐位相同** | ✅ | 兩者**都**與 D 相同（見 §一）—— 一致的一對，不需重建 |

### 品質數字（供第 2、6 道交叉核對）

```
              D              RF-SAFE         delta      movers  worse  stray
WINDOWS 48c   1.226325126    1.215239132    +0.9040%      12      0      0
LINUX   48c   1.226406964    1.217828992    +0.6994%      12      0      0    <- 評分平台
OOS s1                                      +1.1122%   （轉移 123%）
OOS s2                                      +1.2265%   （轉移 136%）
```

⚠️ **Linux 只實現 Windows 增益的 77.4%**，比 `mix` 的 95% 折扣更重。定價要用 Linux 那欄：
在評分平台上過 0.30% 門檻的下限是 `f_eff ≈ 1.9`，而 L308 量到的 f 是 **2.38~2.84** ⇒ 有餘裕。

---

## 四、你報告裡五個未結項的現況

| | 項目 | 現況 |
|---|---|---|
| **§A** | 截止日 08-28 vs 08-31 | **已解，但要看清楚**：延期是**按題目分**的。ABC 通用那份 PDF 寫 08-28 17:00 且**至今沒有更新**；主辦澄清信明講延到 **08-31 23:59 只適用 Problem C**，我們是 C。逐字證據在 `DEADLINE_PRIMARY_EVIDENCE.md`。⚠️ **`C_QA_20260827.pdf` 通篇沒有任何日期陳述**，所以 08-31 目前**只靠那封信**撐著 |
| **§B** | median 漂移敏感度 | **已補**：`SHIP_DECISION_2026-08-28.md` 有逐 `f_eff` 的 NET 表，L313 §4 再加上 Linux 的 77.4% 折扣 |
| **§C** | beta 資料檔沒進 git | **仍然沒有**（`beta_2026-08-16/`、`beta_2026-08-23/`、RFSAFE 的 tar 都 untracked）。不影響送件有效性，但你的顧慮成立 |
| **§D** | `op_wrapper.py:1709` 絕對路徑 | ✅ **RF-SAFE 已修掉**（見 §二第 1 項） |
| **§E** | Case B venv 沒端到端跑過 | ✅ **已跑，PASS** —— 見下 |

### §E 結案：L341 Case B lane

全文 `L341_CASEB_VENV.md`，log `l341_caseb.log`。

```
Stage 1  全新 venv、無系統套件、python 3.14.4
         pip install -r requirements.txt      exit 0, 4m11s
         torch 2.13.0+cu130 / numpy 2.5.2 / scipy 1.18.1 / shapely 2.1.2
         matplotlib 3.11.1 / tqdm 4.70.0 / requests 2.34.2

Stage 2  官方指令跑在那個 venv 裡        105s, exit 0
         Tests 100 | Feasible 100 | Avg Cost 1.3170 | Avg Runtime 0.75s
         total 對錨 |d| = 2.220e-16、worst |dcost| = 0.000e+00（逐案逐位相同）
         scipy: system、bundled-first OK（沒有就地編譯）
         L117 LINUX-VERIFY [final/l341_caseb]: PASS
```

**這比評分機更嚴格**：報告 B 指向 py 3.13，我們跑 **3.14.4**（wheel 供給更稀薄）。
3.14 解得開 ⇒ 3.13 幾乎必然解得開，反過來不成立。

⚠️ 但請注意**這道 lane 不是 RF-SAFE 的閘**：`requirements.txt` 在 D 與 RF-SAFE 之間
**逐位相同**，所以曝險在 **D 上傳的那一刻就已經承擔了**。它不可能是「先別傳 RF-SAFE」
的理由 —— 失敗的話是**兩顆包一起要修**。

⚠️ 它也**沒有**驗證 48c：WSL 是 32 核，在 `≥40` 閘**之下**，所以 RF-SAFE 在該 lane 中
是惰性的、與 D 逐位相同（那是設計上該有的結果，順便獨立再確認一次閘的惰性）。
**48 核的 +0.6994% 是 L313 lane 4 的結果，不是這道的。**

---

## 五、`C_QA_20260827.pdf` 裡影響這次審查的四條

| | |
|---|---|
| **A21** | Final 用**與 beta 完全相同的 hidden set** ⇒ 所有基於 beta median 的定價仍然有效 |
| **A20** | 「你送的 `op_wrapper.py` 就是評估時用的那份，可以自由修改」⇒ 我們的做法合法 |
| **A25 / A27 / A29** | 「**requirements.txt 非空時，我們的 pipeline 一律建 venv**」+ A17「beta 有 submissions 因環境問題壞掉、被重跑」⇒ 這正是 §E 非跑不可的理由。而**beta 送的是空的 `requirements.txt`** ⇒ venv 是 **beta 之後新增的曝險**，不是繼承的 |
| **A27 的矛盾** | A27 說要 **pinned versions**，`l246` 有條規則反過來要求**不要釘**（來自報告 B 75-76）。A27 回的是 Q27「Recommendation for **Numba**」，範圍是那隊的 Numba/llvmlite 問題；我們不出 Numba，而 `>=` 在 py3.14 實測解得開 ⇒ **不動 `requirements.txt`** |

---

## 六、要獨立複驗的話

```bash
# 身分
tar xzOf cadc1075.tar.gz cadc1075/op_wrapper.py | md5sum
#   62db6ee4569b31ddc8c546ccf3e7cd0b

# 合規 20 條
python l246_compliance.py build_submission.RFSAFE/cadc1075.tar.gz

# Case B venv lane（WSL；從 Git Bash 叫要加 MSYS_NO_PATHCONV=1）
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l341_caseb_venv.sh

# Linux 五車道
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l313_wsl_rfsafe.sh
```

🚨 **在 Windows 上複驗的人：先把 msys 放進 PATH，否則你會拿到一個看起來正常的錯答案。**

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
```

沒設的話：包會就地編譯 → `cc1plus.exe` 載不到自己的 DLL → `g++` **回 1 且完全沒有錯誤訊息**
→ 整包無聲退回 Python SA → 印出 **`Total Score: 10.0000` 配 `Feasible: 100/100`**
（那個 10.0000 是 SA 的 feasible 上限 9.999999 被 `%.4f` 進位，**不是全案不可行**）。
`ICCAD_CXX` 修不好這件事 —— 壞的是 `cc1plus` 不是 driver，只有 PATH 有效。
Linux 評分機不受影響（走 bundled ELF，根本不編譯）。

---

## 七、上傳步驟與風險界線

1. 資料夾 = 主辦給每隊的 **`final_test_submission`**（Problem B/C 走 Google Drive）。
   ⚠️ 是 **Final** 不是 Beta，是 **cadc1075** 不是別隊的 `cadb1036`（曾傳錯一次）。
2. **檔名必須逐字是 `cadc1075.tar.gz`。** Chrome 重新下載會加 ` (1)`。
   **主辦白紙黑字寫「檔名錯誤不接受重交」。**
3. **不要再壓縮一層** —— guidelines 明講 "no additional compression is required"。
4. **確認資料夾裡只剩這一顆 tar。** 舊的 D 若還在旁邊，評分機挑哪一顆是未定義的。
5. 上傳後下載回來對 `op_wrapper.py` 的 md5 = `62db6ee4569b31ddc8c546ccf3e7cd0b`。

**風險界線：D 和 D+RF-SAFE 插在同一格 rank 2，RF-SAFE 不換名次**，它買的是餘裕
（被擠下 rank 2 的餘裕 1.49% → **2.10~2.41%**，對第一名的差距 1.92% → **1.00~1.32%**）。
⇒ **任何一步出問題就守 D**（已在 Drive、已驗過、已知良品）。
凍結時點 **08-30 23:59**，過了就不要再動。
