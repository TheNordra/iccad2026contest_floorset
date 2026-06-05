# ICCAD 2026 Alpha Test（**Problem C**）— 提交規範 + 開發環境指南

> **來源**：官方兩份 PDF —《Alpha Test Submission Guideline (ABC)》與
> 《2026 CAD Contest Competition Host User Manual》，內容已轉錄整理；
> 另含本隊 **2026-06-05 VM 實測探查 + Phase 1 smoke test**（標註「實測」）。
>
> **🟦 本隊題目 = Problem C → 繳交走 Google Drive，不是 VM。** 詳見第 2 節。

---

## 📅 1. 時程與通用提交規則（來源：Submission Guideline）

- **Alpha Test 截止**：2026 年 6 月 12 日（週五）**17:00 (GMT+8)**。
- **逾時不收**：系統會準時關閉，正在進行的上傳也會被中斷 → **強烈建議提早上傳**。
- **僅接受指定方式**：不開放雲端（cloud）提交，**不接受 email 提交**。
- ⚠️ **因檔名錯誤而要求重新提交者「不予受理」** → 檔名與放置位置務必一次到位。
- 提交前請詳閱官方 **Q&A 與題目說明**中的個別規則。

### 各題提交管道

| 題目 | 提交管道 | 放置位置 |
|------|----------|----------|
| Problem A | 競賽 VM（SSH） | VM 家目錄下的 `alpha_test_submission/` |
| **Problem C（本隊）** | **官方提供的 Google Drive 連結** | **Drive 內的 `alpha_test_submission/`** |
| Problem B | 官方提供的 Google Drive 連結 | Drive 內的 `alpha_test_submission/` |

---

## 📤 2. 本隊繳交方式：Google Drive（Problem C）

官方為每組建立了名為 `alpha_test_submission/` 的資料夾（透過官方提供的 Google Drive
連結存取）。**把這個階段要提交的所有檔案放進對應的 submission stage 資料夾即可。**

- 三個 submission stage 資料夾（**不可刪除或更名**）：`alpha_test_submission/`、
  `beta_test_submission/`、`final_test_submission/`。
- 目標資料夾內**不需**額外壓縮（no compression）。
- ⚠️ **本隊是 Problem C** → 繳交 = 上傳到 **Google Drive**。
  **不要**把檔案留在開發 VM 上以為那樣算繳交（那是 Problem A 的路徑）。

---

## 🔌 3. 開發/測試主機 VM（本隊已確認）

> ℹ️ 此 VM 已確認為本隊競賽主機（《Host Manual》所述的 competition host），綁定 IP
> 白名單生效中，供開發 / 編譯 / 測試。**Problem C 的繳交仍走 Google Drive，此 VM 不是
> 繳交端，也不是評分端**（理由見第 4 節：VM 全裸 + 無網路，不可能在其上評分）。

- **Host IP**：`140.110.214.90`　**SSH 帳號**：`cada1090`
- **密碼**：初始 `a9c356385`，**本隊已用 `passwd` 改過**；新密碼自行保管，**勿寫入版控檔**。
- **連線白名單**：目前僅允許從綁定 IP（`140.124.72.26`）連入，其他來源被防火牆阻擋。
  - 實測：本機 outbound IP 為 `125.229.238.23` 時 port 22 仍可連 → 白名單目前允許。
- **登入 shell = `/bin/tcsh`**（實測）。寫複雜指令時 `2>/dev/null` 等 bash 語法會報
  "Ambiguous output redirect"；自動化請明確用 `... | bash` 或 `bash -c`。
- **連線方式**：SSH（手冊以 MobaXterm 示範；任何 SSH client 皆可）。自動化批次連線建議用
  SSH 金鑰（password auth 無法用管線餵入）。

### IP 變更流程（來源：Guideline）
- 若需更換綁定 IP，請將**有效且固定（static）的 IP** 寄到 `cad.contest.iccad@gmail.com`。
- 設定需**數個工作天**，逾時申請不受理 → 請**盡早測試連線**。

---

## 🔬 4. VM 實測探查重點（2026-06-05）

| 項目 | 實測結果 | 影響 |
|------|----------|------|
| 主機 / 家目錄 | host `iccad010`，home `/project/cad10/cada1090` | 三個 submission 資料夾已存在且**空** |
| **外部網路** | ❌ **DNS 不通**（pypi / pythonhosted / huggingface 皆 `Name or service not known`），無 proxy、無內部 PyPI mirror | **`pip install` 完全無法用；HuggingFace 資料集無法下載** |
| Python | 系統 `python3` = **3.6.8**（另有 python2）；`module load python/3.12.x` 可用但同樣**全裸** | — |
| 已裝套件 | **無** numpy / torch / scipy / shapely / pandas / requests；pip 僅 **9.0.3** | 官方 evaluator（需 torch+numpy+shapely）**無法直接在 VM 跑** |
| 編譯鏈 | **g++ 8.5.0-24**、**cmake 3.26.5**、glibc 2.28（實測） | C++ solver 可直接編譯 |
| 算力 | **8 cores**、**RAM 實測 ~251 GB**（手冊寫 128 GB）、磁碟 ~134 GB 可用 | portfolio 並行充裕 |
| EDA 工具 | `module avail` 列出完整商用套件（Cadence / Synopsys / Siemens / Mathworks / Agilent / Arm / klayout…），遠多於手冊列的 10 套 | 本題只需 g++ |

### ✅ Phase 1 smoke test 結果（C++ solver 在 VM 上）

把 `constructive.cpp` + 三個序列化輸入 scp 到 VM，用 VM 的 g++ 8.5 編譯並執行：

| case | 編譯 | 執行時間 | 輸出 | 與本地比對 |
|------|------|----------|------|-----------|
| — | g++ 8.5 `-O3 -std=c++17` → **rc=0, 3.6s, 無 warning** | | | |
| n=21 | | **5 ms** | 21 blocks ✓ | METRICS **逐位元相同** |
| n=71 | | **53 ms** | 71 blocks ✓ | METRICS **逐位元相同** |
| n=120 | | **114 ms** | 120 blocks ✓ | METRICS **逐位元相同** |

→ **結論**：提交的核心產物（C++ solver）在競賽 OS/編譯器上能 build、能跑、且輸出與本地
（msys g++ 13）**完全一致** → 本地調參＝VM/評分結果，跨編譯器確定性成立。VM 驗證已足夠。

---

## 🧰 5. 軟體環境細節（來源：Host Manual，本隊實測補充）

### 🟢 編譯工具鏈（Ready，實測）
- **g++ 8.5.0-24** + **cmake 3.26.5** + glibc 2.28。
- `optimizer_claude.cpp` / `constructive.cpp` 可直接 `g++ -O3 -std=c++17` 編譯，無須額外設定。

### 🧩 商用 EDA（`module load <tool>`）
手冊列出 Innovus 21.19 / Design Compiler 2025.06 / Calibre / VCS / Verdi / IC Compiler2 等；
實測 `module avail` 還有 genus / conformal / xcelium / spectre / quantus / pegasus / primetime /
formality / questasim / tessent / catapult / matlab / klayout / java / python 3.12 …
（本題 FloorSet 用不到，列此供參考。）

### 🔴 Python（全裸 + 離線，實測）
- 系統 Python 3.6.8、pip 9.0.3，**未裝任何科學套件**；`module load python/3.12` 亦同。
- **無外部網路** → pip 無法安裝、HuggingFace 無法下載（見第 4 節）。

---

## 🚀 6. Action Plan（依實測修正）

> ⚠️ **舊版「在 VM `pip install torch`」的計畫已作廢** —— VM 無網路、pip 連不到任何 index，
> 離線唯一途徑是把 **cp36 manylinux wheels** 手動 scp 過去再 `pip install --no-index`，
> 成本高且有 py3.6 相依風險。**結論：要在 VM 上跑的東西，走純 C++ 路線最穩。**

- ✅ **VM-safe 路線（建議）**：`constructive.cpp` 純 C++、**不需 torch/numpy** → 直接
  `g++ -O3 -std=c++17 -o constructive constructive.cpp` 即可在 VM 編譯執行（已驗證，第 4 節）。
- ❌ **GNN / Python evaluator 路線**：需 torch+numpy+shapely+dataset，VM 上無法直接取得；
  且本隊最佳解本就不靠 GNN。如某次真的要在 VM 重現官方評分，唯一途徑是離線 wheel + 搬 dataset
  （非必要，因 VM 並非評分環境）。
- 提交產物本身請確保能在**主辦的評分環境**正確執行（Problem C 由主辦從 Google Drive 取件評分）。

---

## ✅ 提交前快速檢查清單（Problem C）

- [ ] 提交檔案已上傳到 **Google Drive** 的 `alpha_test_submission/`（**未**更名、**未**壓縮）。
- [ ] 檔名正確無誤（**錯了不能重交**）。
- [ ] ⚠️ 沒有「只把檔案留在 VM」就以為已繳交（Problem C 繳交在 Google Drive）。
- [ ] C++ 產物可用 g++ 8.5 編譯執行（已於 VM 驗證，輸出與本地一致）。
- [ ] 提交內容不依賴 VM 上沒有的套件（torch/numpy 等）；確認在主辦評分環境可跑。
- [ ] 已在本地保留所有提交檔副本。
- [ ] 已於 **2026-06-12 17:00 (GMT+8)** 前完成上傳（提早上傳，勿卡關閉時點）。
