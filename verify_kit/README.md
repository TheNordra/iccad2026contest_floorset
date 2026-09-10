# 驗收套件 — cadc1075 Final

給**我方（驗收方）**用。刻意不依賴組員 `ship_final/` 底下的任何分析腳本，
所有數字都自己算，不採信包裡或交件說明宣稱的任何值。

```
verify_kit/
├── verify_final.sh      驅動：佈置評分機形狀的目錄 → 跑官方指令 → 判定
├── judge.py             判官：identity / score / diff / logcheck
├── l246_compliance.py   組員的 20 條逐條規則檢查（附官方文件行號），原封搬過來當第二意見
└── README.md            本檔
```

## 0. 先決條件（GPU 機的 WSL2）

* FloorSet checkout 一份（要有 `LiteTensorDataTest/`、八個 loader `.py`、
  `iccad2026contest/iccad2026_evaluate.py`）
* `python3`（評分機是 **3.13**）、`numpy` / `scipy` / `shapely` / `torch`
* glibc **≥ 2.38**（`bin/constructive_linux` 要它）

## 1. 用法

```bash
# 星期日拿到新包後，一行搞定
./verify_final.sh \
  --tar  /path/to/cadc1075.tar.gz \
  --repo /path/to/FloorSet \
  --ref  /path/to/results_D_ref.json \
  --expect <組員宣稱的 48c Linux total> \
  --md5    <組員宣稱的 op_wrapper.py md5>
```

先跑一次 **D**（現在 Drive 上那顆）產出 `results_D_ref.json` 當基準：

```bash
./verify_final.sh --tar build_submission.D/cadc1075.tar.gz --repo . \
                  --expect 1.2264069637381392 --runs 2
cp vk_work/cadc1075/results_1.json ~/results_D_ref.json
```

只驗檔案不跑 solver：加 `--skip-run`（幾秒鐘，可在任何機器上做）。

驗**評分機真正走的那條路**（Section 2 Case B，乾淨 venv + `pip install -r
requirements.txt`）：加 `--venv fresh`。這條會下載 torch，很慢，但它是
**唯一**能回答「評分當天 pip 解到的版本組合能不能跑」的做法——
requirements 全是 `>=` 下限，沒有人驗過。

## 2. 判定門檻（事前註冊，看到組員數字之前就訂死）

| # | 關卡 | 通過條件 | FAIL 代表 |
|---|---|---|---|
| G1 | 身分 / 結構 | 8 成員、0 vendor、`op_wrapper.py` md5 == 宣稱值、`op_src` == `op_wrapper`、無 CRLF、無多餘 `.py` | 交件方與驗收方拿到的不是同一顆 |
| G1b | 絕對路徑 | 0 條 | 官方 §4 checklist 明列，可能 DQ |
| G1c | requirements | 非空、無 `==` 釘死、每個非 stdlib import 都宣告 | 缺 scipy = LP 靜默關閉，**−5.4% 無聲蒸發** |
| G2 | 20 條逐條規則 | 19/20（唯一已知 FAIL = msys64 路徑）；**移除後應為 20/20** | 見 G1b |
| G3 | 分數 | feasible **100/100**、total == 宣稱值（±1e-9）、無 cost > 9.99 | >9.99 = binary 沒跑起來，全面沉到 SA |
| G3b | LP 活著 | `lp_stats` 行數 == 宣稱值（D 是 **71**） | 0 行 = scipy 沒到位；100 行 = 閘沒生效 |
| G4 | log | 無 SA fallback、無 traceback、無 GLIBC / FileNotFoundError | 靜默失效的三種形狀 |
| G5 | determinism | 兩輪 cost **與** positions 逐位相同 | 包裡混進了讀時鐘的東西 |
| G6 | 對 D 0-regression | **0 案變差**，總分不變差 | 有退步就要逐案定價，不能只看總分 |

**額外規則**

* `bin/constructive_linux` 的 md5 **若改變**，必須附重建證據，且
  `strings` 要查得到 source 用 `getenv()` 讀的每一個 `ICCAD_*`。
  若 md5 未變，則 `constructive.cpp` 也必須逐位未變——
  「新 cpp + 舊 ELF」在 Windows 上結構性看不見，而且失敗是**靜默**的。
* 跨平台差異**不可當 FAIL**：shape LP 高度退化，Windows scipy 1.15.3 與
  Linux 1.18.0 會落在同一個 LP 的不同最佳解（L119），D 實測 8/100 案不同、
  總分差 0.0067%。**計分的是 Linux 那個數字。**
* 環境裡殘留任何 `ICCAD_*` 就會量到不是出貨組態的東西——腳本會自動剝除，
  但不要在同一個 shell 裡先跑分析腳本再跑這支。

## 3. 交件清單（請組員一起給）

1. `cadc1075.tar.gz`
2. 期望的 **48c Linux total**（給 `--expect`）
3. `op_wrapper.py` 的 **md5**（給 `--md5`）——tar md5 不可重現，別看它
4. `lp_stats` 期望行數
5. **vs D 的逐檔 diff**（哪些檔動了、為什麼）
6. 她自己的 Linux 五車道 log
7. kill switch 名稱（出事時要能一鍵退回）
8. `beta_2026-08-16/beta_evaluation_results.json` 與 `beta_2026-08-23/` —
   缺這兩份，**我方驗得了品質、驗不了名次**

## 4. 已知基準

| | D（2026-08-26 已上傳） |
|---|---|
| 大小 | 408,795 B，8 成員，0 vendor |
| `op_wrapper.py` md5 | `1c326784de7cd9246cd1f380e2842668` |
| `bin/constructive_linux` md5 | `bc9912072cd97b45b47a03adec7170ce`（自 L137 起未變） |
| `constructive.cpp` md5 | `e2c7b2f418ef2b70b6bff99f7adfbd37`（自 2026-08-19 未動） |
| 48c **Linux** total | `1.2264069637381392`，feasible 100/100，LP 跑 71 |
| 48c **Windows** total | `1.2263251265`（8 案不同，**正常**） |
| scipy 缺席時 | `1.2589744529416786`，feasible 100/100，不炸（rank 4 地板） |

`judge.py` 的 `EXPECT` 表也存著同一組值；換基準時只改那裡。

## 5. 工具自檢紀錄

本專案的紀律是「量測工具要先在**已知答案**的輸入上驗過，輸出才算證據」。
本套件已驗：

* `judge.py score results_L136_48c_anchor.json` → `1.2284738198320346`，
  與 CLAUDE.md 記錄的 L136 48c Windows 值**逐位相同** ⇒ 加權公式正確
* `judge.py diff X X --mode determinism` → 100/100 ⇒ 比對器不會假通過
* `judge.py identity` 對 D 的絕對路徑判定，與 `l246_compliance.py`
  **獨立得到同一條 FAIL（`op_wrapper.py:1709`）** ⇒ 兩把尺互為對照
