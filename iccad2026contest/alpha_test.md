# ICCAD 2026 Alpha Test（**Problem C**）— 提交規範 + 開發環境指南

> **來源**：官方兩份 PDF —《Alpha Test Submission Guideline (ABC)》與
> 《2026 CAD Contest Competition Host User Manual》，內容已轉錄整理於本檔。
> 另含本隊對開發主機的實測探查（標註「本隊實測」）。
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

- 三個 submission stage 資料夾（**不可刪除或更名**）：
  1. `alpha_test_submission/`
  2. `beta_test_submission/`
  3. `final_test_submission/`
- 目標資料夾內**不需**額外壓縮（no compression）。
- ⚠️ **本隊是 Problem C** → 繳交 = 上傳到 **Google Drive**。
  **不要**把檔案留在開發 VM 上以為那樣算繳交（那是 Problem A 的路徑）。

---

## 🔌 3. 開發/測試主機 VM（本隊已確認）

> ℹ️ 此 VM 已確認為本隊競賽主機（《Host Manual》所述的 competition host），目前
> **綁定 IP 白名單生效中**，供開發 / 編譯 / 測試使用。
> 注意：官方《Guideline》把「VM 家目錄繳交」明確指派給 **Problem A**；
> **本隊 Problem C 的繳交仍走 Google Drive（見第 2 節）**，此 VM 不是繳交端。

- **Host IP**：`140.110.214.90`
- **SSH 帳號**：`cada1090`
- **初始密碼**：`a9c356385`（⚠️ 首次登入後請立即用 `passwd` 修改）
- **連線白名單**：目前僅允許從綁定 IP（`140.124.72.26`）連入，其他來源會被防火牆阻擋。
- **連線方式**：SSH。手冊以 **MobaXterm** 示範（Session → SSH → 輸入 VM IP → 輸入帳密），
  任何支援 SSH 的工具皆可。

### IP 變更流程（來源：Guideline，原列於 Problem A 段；如本隊 VM 受 IP 綁定亦適用）
- 若需更換綁定 IP，請將**有效且固定（static）的 IP** 寄到 `cad.contest.iccad@gmail.com`。
- 設定作業需**數個工作天**，逾時申請不受理 → 請**盡早測試連線**，遇問題立即聯絡。

---

## 🖥️ 4. 硬體與作業系統（來源：Host Manual，本隊實測一致）

| 項目 | 規格 |
|------|------|
| CPU | Intel® Xeon® Gold 5320H（每台 VM **8 cores**） |
| Memory | 128 GB |
| GPU | **無**獨立顯示卡 |
| OS | RedHat 8 |

> 純 CPU 傳統運算伺服器，無深度學習專用環境。

---

## 🧰 5. 軟體環境（來源：Host Manual + 本隊實測）

### 🟢 編譯工具鏈（Ready）

- **gcc / g++ 8.5.0-24**（el8），**glibc 2.28**，並提供 **cmake**。
- 影響：C++ 後處理優化器（`optimizer_claude.cpp`、`constructive.cpp`）可直接以
  `g++ -O3 -std=c++17` 編譯，**無須額外設定**。

### 🧩 商用 EDA 工具（透過 `module load` 載入）

手冊列出已安裝的工具與其載入方式（載入後即可直接使用該工具指令）：

| 軟體 | 版本 | 載入指令 |
|------|------|----------|
| Innovus | 21.19.000 | `module load innovus; innovus -stylus` |
| Conformal | 24.10.100 | `module load conformal; lec` |
| Design Compiler | 2025.06 | `module load dc; dc_shell-t` |
| XCelium | 24.09.006 | `module load xcelium; ncverilog`（或 `xrun`） |
| Calibre | aoj/2025.2_14.11 | `module load calibre; calibre` |
| VCS | 2025.06 | `module load vcs; vcs` |
| Verdi | 2025.06 | `module load verdi; verdi`（或 `nWave`） |
| Library Compiler | 2024.09-sp2 | `module load lc; lc_shell` |
| Laker | 2024.12 | `module load laker; laker` |
| IC Compiler2 | 2024.09-sp2 | `module load icc2; icc2_shell` |

> 本題（FloorSet floorplanning）主要只會用到 g++；上列商用工具列出供參考。

### 🔴 Python / 深度學習環境（需手動設定）

- 預裝 **Python 2 與 3.6**；**未**預裝 `conda` 與 `torch`，環境乾淨。
- **無 GPU** → 任何 ML 推論都必須走 CPU，需特別留意是否會 timeout。
- ⚠️ **現況**：本隊最佳解 `constructive.cpp`（Total Score **1.7045**，~0.16s/case）為
  **純 C++、不需 torch**。下方 Python/torch 設定**僅在仍要走 GNN 路線**
  （`optimizer_claude.py` + `floorplan_gnn.pth`）時才需要。

---

## 🚀 6. 待辦與防呆（Action Plan）

### Step 1：確認真實 Python 版本
```bash
python3 --version
```

### Step 2：（僅 GNN 路線）安裝 CPU-only PyTorch
無 GPU，切勿安裝含 CUDA 的預設版；安裝最輕量的純 CPU 版：
```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --user
```

### Step 3：（僅 GNN 路線）程式碼 Device Fallback（極重要）
若未加 `map_location`，模型在 VM 上找不到 CUDA 會直接 `RuntimeError` 崩潰。
推論程式碼務必包含自動降級邏輯：
```python
import torch

# 1. 裝置自動偵測：本地端用 GPU，VM 自動降級為 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 載入權重時強制映射到當前可用裝置
model.load_state_dict(torch.load("floorplan_gnn.pth", map_location=device))
model.to(device)
model.eval()
```

---

## ✅ 提交前快速檢查清單（Problem C）

- [ ] 提交檔案已上傳到 **Google Drive** 的 `alpha_test_submission/`（**未**更名、**未**壓縮）。
- [ ] 檔名正確無誤（**錯了不能重交**）。
- [ ] ⚠️ 確認沒有「只把檔案留在 VM」就以為已繳交（Problem C 繳交在 Google Drive）。
- [ ] C++ 已在開發 VM 上以 g++ 8.5 重新編譯驗證；若走 GNN 路線，torch 為 CPU 版且已加 `map_location`。
- [ ] 已在本地保留所有提交檔副本。
- [ ] 已於 **2026-06-12 17:00 (GMT+8)** 前完成上傳（提早上傳，勿卡關閉時點）。
