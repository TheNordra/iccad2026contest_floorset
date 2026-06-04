# ICCAD 2026 Alpha Test - 官方 VM 環境探查與建置指南

## 📌 1. 虛擬機 (VM) 連線資訊
- **Host IP**: `140.110.214.90`
- **SSH 帳號**: `cada1090`
- **初始密碼**: `a9c356385` (⚠️ **首次登入後請立即使用 `passwd` 指令修改**)
- **網路白名單**: 僅允許從綁定 IP (`140.124.72.26`) 進行連線，否則會被防火牆阻擋。
- **Alpha Test 截止時間**: 2026 年 6 月 12 日 (週五) 17:00 (GMT+8)

## ⚠️ 2. 系統目錄規範
帳號家目錄下的這三個資料夾為大賽評分系統自動抓取用的對接端口，**絕對不可刪除或更名**：
1. `alpha_test_submission/`
2. `beta_test_submission/`
3. `final_test_submission/`

---

## 🖥️ 3. 系統環境探查結果 (OS: RedHat 8)
經由終端機實測與大賽手冊比對，官方 VM 為**「純 CPU 傳統運算伺服器」**，無深度學習專用環境。

### 🟢 C++ 編譯環境 (Ready)
* **狀態**: 支援良好。
* **版本**: `gcc` / `g++` 8.5.0 (Red Hat 8.5.0-24)。
* **影響**: C++ 版本的後處理優化器（如 `optimizer_claude.cpp`）可直接在此環境使用 `g++ -O3` 等指令進行編譯，無須額外設定。

### 🔴 GPU 硬體 (No GPU)
* **狀態**: 無獨立顯示卡。
* **硬體**: Intel® Xeon® Gold 5320H (8 cores), 128GB RAM。
* **影響**: 神經網路模型（GNN）上傳至此 VM 後，**必須全數依賴 CPU 進行推論計算**，需特別注意推論時間是否會導致 Timeout。

### 🔴 Python / 深度學習環境 (Manual Setup Required)
* **狀態**: 環境極度乾淨，未預裝 `conda` 與 `torch`，Python 官方標示為 3.6。
* **影響**: 無法直接執行現有的 PyTorch 訓練或推論腳本，需手動透過 pip 安裝輕量化套件。

---

## 🚀 4. 關鍵待辦事項與防呆策略 (Action Plan)

### Step 1: 確認真實 Python 版本
進入 VM 後，首要任務是確認系統實際的 Python 3 版本，以決定後續套件支援度。
```bash
python3 --version
```

### Step 2: 手動安裝 CPU-only PyTorch
由於沒有 GPU，切勿安裝預設包含龐大 CUDA 驅動的 PyTorch。請在 VM 終端機執行以下指令，安裝最輕量化的純 CPU 版本

```bash
python3 -m pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) --user
```

### Step 3: 實作程式碼 Device Fallback (極度重要)
提交至 `alpha_test_submission` 的 Python 主程式（包含模型推論的區塊），必須實作自動降級機制。若未加上 `map_location`，模型在 VM 上尋找 CUDA 失敗時會直接引發 `RuntimeError` 崩潰。

請確保代碼包含以下邏輯：
```python
import torch

# 1. 裝置自動偵測：本地端用 GPU，VM 測試端自動降級為 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 載入權重時，強制映射到當前可用裝置
model.load_state_dict(torch.load("floorplan_gnn.pth", map_location=device))
model.to(device)
model.eval()
```