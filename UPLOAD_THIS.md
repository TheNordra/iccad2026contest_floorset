# 上傳這一顆 — 2026-08-28

> 決策見 [SHIP_DECISION_2026-08-28.md](SHIP_DECISION_2026-08-28.md)：送 **D + RF-SAFE**。
> 截止 **2026-08-31 23:59 GMT+8**（Problem C 專屬延期）。

## ✅ 要上傳的檔案

```
C:\ICCAD_ml\ship_final\build_submission.RFSAFE\cadc1075.tar.gz
```

| | |
|---|---|
| 大小 | **409,589 bytes** |
| tar 成員 | **8**（無 `vendor/`） |
| **身分：`op_wrapper.py` md5** | **`62db6ee4569b31ddc8c546ccf3e7cd0b`** |
| `op_src.py` md5 | `62db6ee4569b31ddc8c546ccf3e7cd0b`（與 op_wrapper 相同） |
| `constructive.cpp` md5 | `e2c7b2f418ef2b70b6bff99f7adfbd37`（與 D 相同） |
| `bin/constructive_linux` md5 | `bc9912072cd97b45b47a03adec7170ce`（與 D 相同） |
| `README.md` md5 | `8167121dbb2a0bcb05851226a6f6a4e3` |
| `requirements.txt` md5 | `6c59feb458f1f48247373d8b69f401c2` |

🚨 **不要用 tar 的 md5 當身分**——gzip 內嵌 mtime，每次 stage 都會變。**身分一律看
`op_wrapper.py` 的 md5。**

## 上傳步驟

1. **資料夾**：主辦給每隊的 **`final_test_submission`**（Problem B/C 走 Google Drive）。
   ⚠️ 是 **Final** 不是 Beta，是 **cadc1075** 不是別隊的 `cadb1036`（曾傳錯一次）。
2. **檔名必須逐字是 `cadc1075.tar.gz`。**
   ⚠️ Chrome 重新下載會加 ` (1)` ⇒ 變成 `cadc1075 (1).tar.gz`。
   **主辦白紙黑字寫「檔名錯誤不接受重交」。**
3. **確認資料夾裡只剩這一顆 tar。** 舊的 D 若還躺在旁邊，評分機挑哪一顆是未定義的。
4. 上傳後**下載回來、對一次 `op_wrapper.py` 的 md5**：

```bash
tar xzOf cadc1075.tar.gz cadc1075/op_wrapper.py | md5sum
#   要看到 62db6ee4569b31ddc8c546ccf3e7cd0b
```

## 交給驗收方的六項

| # | 項目 | 值 |
|---|---|---|
| ① | 期望 48c Linux total | **1.2178289924684162**，feasible 100/100 |
| ② | `op_wrapper.py` md5 | **`62db6ee4569b31ddc8c546ccf3e7cd0b`** ⚠️ **不是** D 的 `1c326784…`，也不會等於它 |
| ③ | `lp_stats.txt` 行數 | **83**（D 是 71） |
| ④ | vs D 的逐檔 diff | **4 hunks**，全在 `op_wrapper.py`/`op_src.py`：L240 探針（既有、預設 OFF）、msys 那行 + 註解、`_L196_LPGATE`。其餘四檔逐位相同 |
| ⑤ | 五車道 log | L313，全綠 |
| ⑤b | **Case B venv lane** | **L341 PASS** — 全新 venv（py 3.14.4）`pip install` 4m11s 解開、官方指令跑完 **100/100 feasible**、對錨**逐位相同**（worst \|dcost\| = 0）、`scipy: system`、bundled-first。⚠️ D 的 `requirements.txt` 與此**逐位相同**，所以這道閘對兩顆包同時成立 |
| ⑥ | kill switch | `ICCAD_L223_REFINE_HEAVY`、`ICCAD_L231_REFINE_MID`、`ICCAD_LP_GATE`、`ICCAD_SHAPE_LP`、`ICCAD_M80_TIER`、`ICCAD_M67F_TIER5` |

## ⚠️ 在 Windows 上複驗的人必讀

**先把 msys 放進 PATH，否則你會拿到一個看起來正常的錯答案。**

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
```

沒設的話：包會嘗試就地編譯 → `cc1plus.exe` 載不到自己的 DLL（它住在 `lib/gcc/…`，
DLL 在 `ucrt64/bin`）→ `g++` **回 1 且完全沒有錯誤訊息** → 整包無聲退回 Python SA，
印出 **`Total Score: 10.0000` 配 `Feasible: 100/100`**。
那個 10.0000 是 SA 的 *feasible 上限* 9.999999 被 `%.4f` 進位，**不是全案不可行**。

🚨 **`ICCAD_CXX` 修不好這件事**——指到絕對路徑會撞上同一個 `cc1plus` 失敗。
壞的是 `cc1plus` 不是 driver，所以**只有 PATH 有效**。
（`g++ --version` 回 0 **不代表能編譯**：`--version` 只跑 driver，不會叫起 `cc1plus`。）

Linux（評分機）不受影響：走 bundled ELF，根本不編譯。

## 這顆包是什麼（一段話）

出貨包 D，**只改一張 wrapper 表**：`_L196_LPGATE` 從 71 開放到 **83**，多開的 12 個
block count 是 `[38, 40, 56, 76, 79, 81, 94, 95, 107, 108, 114, 120]` —— 那些**增加的
grader 時間塞得進自己到 RF floor 的 slack** 的案子。挑選**完全不看品質**。
`constructive.cpp` 與 `bin/constructive_linux` 逐位不變 ⇒ **沒有重建 ELF**。

驗過的：l246 **20/20**、48c Windows +0.9040% / Linux +0.6994%（各 12 movers、**0 worse**、
0 stray、100/100 feasible）、預設核數下與 D **逐位相同含 positions**、
OOS 兩份互斥樣本 **+1.1122% / +1.2265%**（轉移 123% / 136%）、Linux 五車道全綠。

## 其他七顆同名 tar

| 目錄 | op_wrapper md5 | 是什麼 |
|---|---|---|
| **`build_submission.RFSAFE`** | **`62db6ee4`** | ✅ **上傳這顆** |
| `build_submission.D` | `1c326784` | 前一次上傳、**fallback**（仍有效） |
| `build_submission.MIXD` | `2b795995` | mix 候選（未送，見 SHIP_DECISION §二） |
| `build_submission` | `2b795995` | mix 的 vendor stage |
| `build_submission.NOVENDOR` | `1c326784` | D 的無 vendor 變體 |
| `build_submission.L131FIX` / `.L136FIX` | `2967efb6` | 舊包 |
| `build_submission.MIX` | `b74cadae` | 舊包 |
| `build_submission.SHIPPED.bak` | `ad8c5dcb` | 舊包 |

除了 RFSAFE 與 D 之外，每個目錄都放了 `DO_NOT_UPLOAD.txt`。

## 如果任何一步出問題

**守 D。** 它已經在 Drive 上、已經驗過、是已知良品，整個 f·m 範圍都是 rank 2。
RF-SAFE 的增益是餘裕不是名次（兩者都是 rank 2），所以**不值得為了它冒任何上傳風險**。
凍結時點：**08-30 23:59**，過了就不要再動。
