# 出貨包獨立驗證 Runbook — cadc1075, 2026-08-27

給隊友做**獨立複驗**用。自足：只需要這顆 tar、比賽的 evaluator 與 dataset，
不需要 `ship_final/` 底下任何分析腳本。

---

## 0. 要驗的東西

| | |
|---|---|
| 檔案 | `build_submission.D/cadc1075.tar.gz`（也就是 Drive Final 資料夾裡那顆） |
| 大小 | **408,795 bytes** |
| 身分 md5（`op_wrapper.py`） | **`1c326784de7cd9246cd1f380e2842668`** |
| tar md5 | 不可重現（gzip 內嵌 mtime）——**身分一律看 op_wrapper，不要看 tar** |

**期望結果（48 核 Linux，官方指令）**

```
total  = 1.2264069637381392
Tests  = 100        Feasible = 100/100
LP 實際執行的 block count 數 = 71
```

⚠️ **Windows 上會得到 1.2263251265，8/100 案不同。這是正常的**，不是迴歸：
shape LP 高度退化，Windows scipy 1.15.3 與 Linux 1.18.0 落在**同一個 LP 的不同
最佳解**（L119）。計分的是 Linux 那個數字。

---

## 1. 五分鐘版：不跑,只驗檔案（任何機器都能做）

```bash
tar xzOf cadc1075.tar.gz cadc1075/op_wrapper.py | md5sum
#   要看到 1c326784de7cd9246cd1f380e2842668

tar tzf cadc1075.tar.gz
#   要恰好 8 個成員，且沒有任何 vendor/ 開頭的項目：
#     cadc1075/  cadc1075/README.md  cadc1075/bin/  cadc1075/bin/constructive_linux
#     cadc1075/constructive.cpp  cadc1075/op_src.py  cadc1075/op_wrapper.py
#     cadc1075/requirements.txt

tar xzOf cadc1075.tar.gz cadc1075/requirements.txt
#   要恰好 7 行、全部 >=、含 scipy、torch>=2.5.0
```

規則逐條檢查（20 條，含文件行號）：

```bash
python l246_compliance.py /path/to/cadc1075.tar.gz
```

預期 **19/20 字面通過**。唯一的 FAIL 是 `op_wrapper.py:1709` 的
`C:\msys64\ucrt64\bin\g++.exe`——它在 `if os.name == "nt":` 裡面，POSIX 上不可達，
而且**同一行、同一個守衛在 commit `7f38893`（M73）裡，那顆包已經被實際評分過**。
已知、有守衛、活過一次評分。

---

## 2. 完整版：48 核 Linux 跑官方指令

### 2.1 目錄佈置（評分機的形狀）

evaluator 會 `sys.path.insert(0, parent.parent)`，dataset 的預設路徑是相對於
cwd 的 `../`，所以佈置成：

```
work/
├── LiteTensorDataTest/          ← 比賽的驗證集（101 個 config_* 目錄）
├── litetestLoader.py            ← 以下七個 loader 來自比賽 repo
├── lite_dataset_test.py
├── liteLoader.py
├── lite_dataset.py
├── prime_dataset.py
├── cost.py
├── utils.py
├── visualize.py
└── cadc1075/                    ← tar 解開後的內容
    ├── op_wrapper.py
    ├── op_src.py
    ├── requirements.txt
    ├── README.md
    ├── constructive.cpp
    ├── bin/constructive_linux   ← 記得 chmod +x
    └── iccad2026_evaluate.py    ← 從比賽 repo 複製進來
```

```bash
mkdir -p work && cd work
tar xzf /path/to/cadc1075.tar.gz
chmod +x cadc1075/bin/constructive_linux
cp /path/to/iccad2026contest/iccad2026_evaluate.py cadc1075/
cp /path/to/{litetestLoader,lite_dataset_test,liteLoader,lite_dataset,prime_dataset,cost,utils,visualize}.py .
ln -s /path/to/LiteTensorDataTest .
```

### 2.2 環境（照主辦 Case B 的做法）

```bash
cd cadc1075
python3 -m venv .venv_eval
.venv_eval/bin/pip install -r requirements.txt
```

### 2.3 跑

```bash
cd cadc1075
env -u $(env | grep -o '^ICCAD_[A-Z_]*' | tr '\n' ' ' | sed 's/ / -u /g') \
    ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS=$PWD/lp_stats.txt \
  .venv_eval/bin/python -u iccad2026_evaluate.py --evaluate op_wrapper.py \
  -o results_verify.json
```

🚨 **把環境裡所有 `ICCAD_*` 清乾淨**（上面那行 `env -u` 在做的事）。這棵樹的
分析腳本會設一堆 `ICCAD_*`，殘留一個就會量到不是出貨組態的東西——這是本專案
記錄最多次的無聲失效。

### 2.4 對答案

```bash
python - <<'EOF'
import json, math
d = json.load(open("results_verify.json"))["test_results"]
W = lambda n: math.exp(n / 12.0)
SW = sum(W(r["block_count"]) for r in d)
tot = sum(W(r["block_count"]) * r["cost"] for r in d) / SW
print("total    =", repr(tot))
print("feasible =", sum(1 for r in d if r["is_feasible"]), "/", len(d))
EOF
wc -l lp_stats.txt          # 要是 71
```

| 檢查 | 期望 | 不符代表 |
|---|---|---|
| `total` | `1.2264069637381392` | 見下表 |
| `feasible` | 100/100 | 有案子壞掉，立刻停下來 |
| `wc -l lp_stats.txt` | **71** | LP gate 沒生效或表被換掉 |
| log 裡 `[scipy] source=` | `system` | 見下 |
| log 裡 `SA fallback` | **0 次** | binary 沒跑起來，分數會變 ~10 |

---

## 3. 讀懂不符的情況

| 症狀 | 意義 |
|---|---|
| `total` 差在 1e-12 以內 | 正常浮點，過 |
| **8 案不同、其餘一致** | 正常。退化 LP 的跨平台差異（L119），Windows vs Linux 就是這樣 |
| `lp_stats.txt` 是 **0 行** | scipy 沒裝到 ⇒ LP 整條靜默關閉。`total` 會是 `1.2589744529416786`、仍然 100/100 feasible、**不會炸**——但那是 rank 4 的地板，不是 rank 2 |
| `lp_stats.txt` 是 **100 行** | `ICCAD_LP_GATE=0` 殘留在環境裡 |
| `SA fallback` > 0 | `bin/constructive_linux` 沒跑起來（忘了 chmod +x，或 GLIBC 太舊——它需要 ≥ 2.38） |
| `total` ≈ 10.0 | 同上，而且是全面性的 |

---

## 4. 我們自己跑過什麼（供對照）

**in-set 十支 arm 全綠**（`l238_verdict.py`）：determinism cost/positions 100/100、
gate 精確跑在表列的 71 隻上、depth map 平坦、10 支 arm 都 100/100 feasible、
兩個 REFINE kill switch 各自只動自己那一帶、LP 值 +2.5933%（70 動 / 70 好 / 0 壞）。

**Linux 五車道全綠**（`l238_wsl_final.sh`，`L238_WSL_RC=0`）：

```
LANE 1 LP off    1.2589744529
LANE 2 L147 off  1.2408502669   +1.4396% vs base
LANE 3 shipped   1.2264069637   +2.5868% vs base, feasible 100/100
LANE 4 determinism on Linux     cost 100/100  positions 100/100
LANE 5 gate live                ran on exactly the 71, skipped 29
```

**無 vendor 版另外驗過兩條**（`l244b_wsl.sh`）：
scipy 可用 `1.2264069637381392`（與含 vendor 版**逐位相同**、L117 PASS）；
scipy 被擋 `1.2589744529416786`、feasible 100/100、不炸。

---

## 5. 這顆包是什麼（一段話）

48 核上跑 51 隻確定性的 C++ constructive placer（portfolio），用 baseline-free 的
proxy 逐案挑贏家，然後對其中 71 個 block count 跑一次 shape LP 做形狀合法化。
`_M49_REFINE_BAND` 是 `(60,100,"2") + (100,inf,"2")`，`_L196_LPGATE` 71 開 / 29 關，
`_L157_DEPTH` 平坦 `{1:100}`，route A 關閉，`_L211_POOLDROP` 存在但預設關閉。
兩個 kill switch：`ICCAD_L223_REFINE_HEAVY`、`ICCAD_L231_REFINE_MID`。

投影：NET +5.224% vs 我們的 beta，graded **0.87818**，**rank 2**，
距 rank-2 門檻（0.888187）餘裕 **1.08 pp**。
