# M67 — Beta 送件打包計劃(2026-07-20 定案)

## 背景事實(已驗證,勿重查)

- **Alpha 結果分析(2026-07-20 session)**:alpha 測資 = 本地 LiteTensorDataTest **逐位相同**(100/100 案 gap 誤差 0.0);我們 = cadc1075 = Rank 3 官方 1.0286(送件版 = M10, raw 1.4528);**RF 真實**,官方分 = raw × 逐案 RF,我們 cost-加權 RF = **0.7081 ≈ floor**;反推 cross-submission median ≈ 3.2× 我們逐案 runtime。細節見 memory `[[alpha-results-2026-07]]` 與 `Downloads/cadc1075_results.json`、`cadc1075.xlsx`、`C_Alpha_Top5.csv`。
- **Beta guidelines**(`Downloads/beta_submission_guidelines_problemC.txt`):alpha 只是 pipeline exercise;Beta 跑**真 hidden cases**,每隊獨佔 **48-core ICELAKE + A100 80GB + 128GB RAM**;格式錯誤 = DQ。deadline **2026-07-31 17:00 GMT+8**。
- **Q&A**(`Downloads/C_QA_20260618.pdf`):evaluator import op_wrapper 一次、逐案呼叫 `solve()`;案內 multiprocessing 明文允許;官方範本(附錄 A)只是 executable 包法示例,逐案 subprocess timeout=60s;area ±1% 對稱硬約束;preplaced 硬 / boundary 軟。
- **架構決策**:op_wrapper.py = 我們的 Python portfolio wrapper 直接合規化,**不走 PyInstaller**(逐案 spawn+import 稅 0.5-3s 會毀 RF;alpha 已證 Python-module 路線在官方 harness 可跑)。C++ 以 bundled Linux binary 優先 + 現場編譯 fallback + SA 最後防線(M48 三層安全網保持)。
- **戰略不變**:送件內容 = M51 shipped 逐位(1.3265 local);品質軸/L1/L3/in-window LP 全部不動;Beta 回傳資料將是第一份真 hidden-set RF median 校準源,Final 前才重評 M54/M62。

## 送件物流(官方公告 `Downloads/beta_test_submission_ABC.pdf`,與 guidelines 並行遵守)

- Deadline **2026-07-31 17:00 GMT+8 硬截止**:上傳中斷即失敗、遲交不收 → **提早數天上傳**。
- Problem C 送件方式 = **各隊專屬 Google Drive「beta_test_submission」資料夾**(email 不收)。把 `cadc1075.tar.gz` 放入對應 stage 資料夾。
- **檔名錯誤不得重交**("Re-submissions due to errors in file naming will not be accepted")→ 命名零容錯:`cadc1075.tar.gz`、解開後 `cadc1075/`、`op_wrapper.py`/`op_src.py`/`requirements.txt` 逐字元照規格;M67-B 的衛生 assert 必須涵蓋命名檢查。
- ⚠️ 行動項(使用者/組員,**現在就確認**):隊伍的 Google Drive beta_test_submission 資料夾連結在誰手上、能否存取——存取問題要找主辦方,處理需時,不能拖到截止前。

## 封包規格(Beta guidelines 硬規定)

```
cadc1075/
    op_wrapper.py      # REQUIRED, entry point, subclass FloorplanOptimizer
    op_src.py          # 完整原始碼副本(op_wrapper 失敗時的備援)
    requirements.txt   # 空檔 0 bytes(shapely/scipy/numpy/torch/numba 官方環境已提供)
    README.md          # 簡短:編譯 fallback 行為說明
    constructive.cpp   # 主 placer 原始碼(現場編譯 fallback 用)
    optimizer_claude.cpp  # SA fallback 原始碼
    bin/constructive_linux   # 預編譯 Linux 靜態 binary(M67-C 產出)
    bin/sa_linux             # SA 預編譯(同上)
```
- 除 op_wrapper.py / op_src.py 外**不得有其他 optimizer .py**(SA wrapper 邏輯併入 op_wrapper.py 單檔)。
- 包內禁止:.pkl、results json、probe 工具、log、絕對路徑字串。
- 評測指令(在 cadc1075/ 內):`python iccad2026_evaluate.py --evaluate op_wrapper.py`。

## Session 分工(依序執行)

### M67-A:編譯鏈與路徑衛生(repo 內改造)
1. `optimizer_constructive.py` 編譯鏈(M48 `_ensure_compiled`):
   - msys 絕對路徑 `C:\msys64\...` gate 到 `os.name == "nt"`(POSIX 跳過該候選)。
   - 新增 **bundled-binary-first**:POSIX 且存在 `<wrapper_dir>/bin/constructive_linux` 時,先 `chmod +x` + `_binary_runs()` 1-block smoke,通過即用免編譯;失敗照走既有編譯鏈(g++→clang++→c++ × -O3→-O2)。Windows 行為逐位不變。
   - SA fallback(optimizer_claude)編譯路徑同樣衛生化;對 optimizer_claude 的 import 改為 lazy try/except(為 M67-B 併檔鋪路),並支援 `bin/sa_linux`。
2. 全檔絕對路徑掃描(`C:\\`、`/home/`、`/Users/`),殘餘一律 gate 或改相對。
3. **Gate**:官方 local eval 逐位 = `results_shipped_m51.json`(總分 1.3264731049、100 案 positions 逐位)+ `regression_suite.py` 全 PASS。commit。

### M67-B:打包器 + 官方佈局 Windows 驗證 ✅ 完成(2026-07-21)
1. 新工具 `make_submission.py`(永不 ship;`stage|verify|all` 三模式,可 import——`build_op_wrapper_text()` 與 m48 variant 共用同一份併檔實作):
   - staging `build_submission/cadc1075/` 依上方封包規格產出(bin/ 缺席容忍 + 警告,M67-C 補)。
   - `op_wrapper.py` = optimizer_constructive.py 內容 + **機械切片併入** optimizer_claude.py。⚠️ 實測 naive 全檔串接會炸:optimizer_claude 的 module-level `_BIN`/`_ensure_compiled`/`MyOptimizer` 會 shadow constructive 側(每 profile 都會跑 SA binary)→ 只切兩塊:**B1** =「Fallback Python SA」區(python_sa_solve + Skyline/skyline_decode/violation_cost + COL_*/BOUNDARY_* 常數)、**B2** = `_serialize_input`+`_parse_output`;SA MyOptimizer/GNN/SA 編譯鏈刻意不併(constructive 從不呼叫)。M67-A 的 try/except lazy import **整塊移除**(非留著失敗——repo 佈局跑 gate 時 sys.path 會撈到 repo 的 optimizer_claude 遮蔽併檔 bug)。合併 assert:compile()、`class MyOptimizer`/三函式 count==1、tail 禁 `_BIN =`/`_ensure_compiled`/`MyOptimizer`/`_GNN`、AST module-level 名字零碰撞。
   - `op_src.py` = op_wrapper.py 逐位副本;`requirements.txt` 0 bytes;README 簡短(英文,三層 fallback 說明)。
   - 衛生 assert:檔案白名單 exact、`.py` 恰兩隻、絕對路徑掃描(`[Cc]:[\\/]|/home/|/Users/`,allowlist 僅 nt-gated msys 候選)、禁 .pkl/.json/.log/.exe、tar 成員名單複驗、bin/* mode 0o755。
2. Windows 官方佈局驗證(`verify` 模式):解 tar → `build_submission/verify/cadc1075/` + evaluator 疊入包內 + 8 隻 loader .py 疊 `verify/`(evaluator `sys.path.insert(parent.parent)` + data_path `../` 力學)+ `LiteTensorDataTest` junction;官方指令 `python iccad2026_evaluate.py --evaluate op_wrapper.py`(env 剝 ICCAD_*)→ **全 100 案逐位 PASS**:total 1.326473104916827、100/100 feasible、costs/gaps/positions 全逐位、零 fallback、eval 187s(avg 1.57s)。gotcha:子行程須 `encoding="utf-8"`(tqdm 輸出撞 cp1252 decode)。
3. `m48_coldstart_dryrun.py opwrapper` variant(預設模式輸出逐字不變):scratch = 送件佈局(op_wrapper.py + 兩 .cpp、無 optimizer_claude.py)→ **四 phase 全 PASS**(cold compile 3 案逐位含案 99 = 1.3083538507526609、垃圾 exe smoke 攔截、bogus ICCAD_CXX 鏈跳過、phase 4 僅剩 msys 候選)。
4. **Gate 全綠**:逐位 ✅ + coldstart 四 phase ✅ + regression_suite 六項 ✅。commit(build_submission/ 進 .gitignore,make_submission.py 進 repo)。

### M67-C:Linux static binary + 三關驗證(需使用者在 GPU 機 WSL2 跑指令)
1. 產 build script:`g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o bin/constructive_linux constructive.cpp`(先試,必要時試全 `-static`)+ md5 + 1-block smoke;`optimizer_claude.cpp` 同。
2. 更新 Linux 驗證包(參考 memory `[[docker-linux-coldstart-verify]]` 的三關結構,改 cadc1075/ + op_wrapper.py 佈局):Tier1 compile+smoke / Tier2 coldstart / Tier3 100 案逐位 vs `results_shipped_m51.json`(容忍 case 84 <2e-9 ULP)。
3. 使用者在 GPU 機 WSL2(Ubuntu-22.04, g++11)執行、貼回輸出;session 判定、將 binary 入包、確認 bundled-binary-first 路徑在 Linux 實際生效(stderr/耗時證據),重跑 M67-B gate。
4. 風險註記:grader glibc 未知 → binary 只是第一層,編譯 fallback 第二層、SA 第三層。

### M67-D:OOS 泛化預檢(訓練集抽樣;獨立,可與 C 對調)
1. 新工具 `m67_oos_probe.py`(永不 ship):從 `floorset_lite` 訓練集抽 N≥100 案(固定 seed;n 分佈 mirror validation 的 20-120 結構),loader 參考 `tree_decode_probe.py`/`m52_phase0_probe.py`;baseline 由 fp_sol label 導出(鏡射 `ContestEvaluator._extract_baseline`),評分用官方 `evaluate_solution` + `target_positions`(嚴格硬約束;勿用 `tree_decode_probe._cost_of`)。
2. 跑 shipped M51 portfolio(default env)→ OOS raw 加權總分 vs in-set 1.3265;band 分解(n>100/mid/small)、feasible 率、SA fallback 率、最壞案清單。
3. 判定(診斷性,不改送件形):OOS ≤ ~1.40 綠;>1.45 報告 overfit 集中在哪個 band/機制。快取斷點續跑(耗時 ~N×1.6s + 開發)。

### M67-E(可選):48c RF 投影更新
- `rf_score_model.py` cores 網格加 48;以 alpha 錨(weighted RF 0.7081、median≈3.2×)校準;確認 48c 下 tier-4/lowcore fail-open→universal、wall=max_i 假設;純 cache 分析,產 Final 決策用投影。不動送件形。

## 等待中(勿在上述 session 內處理)
- **Alpha feedback 確認**(組員):guidelines 寫 "address all issues noted in your Alpha feedback"——「feedback」很可能就是已拿到的 `cadc1075_results.json`+`cadc1075.xlsx` 本體 + guidelines §4 common issues(其中 (e) 檔名 my_optimizer.py 已確認是我們,M67-B 涵蓋)。只需跟組員確認:主辦方有無寄過**其他**個別通知;若無,此項即結案。
- **Alpha 包內容**(組員):確認當時是預編譯 binary 還是現場編譯過關 → 只影響 M67-C 的保險層級評估,雙路徑架構已覆蓋兩種答案。
- 送件上傳動作本身(使用者執行)。

## 附錄 A:官方 op_wrapper.py 範本(2026-06-18,參考用——我們不採 executable 路線)

```python
# op wrapper
import json
import os
import subprocess
from pathlib import Path
from iccad2026_evaluate import FloorplanOptimizer

class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        base_dir = Path(__file__).resolve().parent
        env_bin = os.environ.get("MY_OPT_BIN")
        candidates = []
        if env_bin:
            p = Path(env_bin)
            candidates.append(p if p.is_absolute() else (base_dir / p))
        candidates.extend([
            base_dir / "dist" / "my_optimizer" / "my_optimizer",  # PyInstaller --onedir
            base_dir / "my_optimizer",                            # PyInstaller --onefile
            base_dir / "bin" / "my_optimizer",                    # optional layout
        ])
        self.bin_path = next((p for p in candidates if p.exists()), candidates[0])
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Optimizer executable not found: {self.bin_path}")
        if not os.access(self.bin_path, os.X_OK):
            raise PermissionError(f"Optimizer is not executable: {self.bin_path}")

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        payload = {
            "block_count": int(block_count),
            "area_targets": area_targets.tolist(),
            "b2b_connectivity": b2b_connectivity.tolist(),
            "p2b_connectivity": p2b_connectivity.tolist(),
            "pins_pos": pins_pos.tolist(),
            "constraints": constraints.tolist(),
            "target_positions": target_positions.tolist() if target_positions is not None else None,
        }
        proc = subprocess.run(
            [str(self.bin_path)], input=json.dumps(payload), text=True,
            capture_output=True, timeout=60, check=True,
        )
        if not proc.stdout.strip():
            raise RuntimeError(
                f"Optimizer produced empty stdout. stderr: {proc.stderr.strip()}")
        data = json.loads(proc.stdout)   # expects {"positions": [[x,y,w,h], ...]}
        if "positions" not in data:
            raise ValueError(
                f"Optimizer JSON must contain 'positions'. Got keys: {list(data.keys())}")
        return [tuple(map(float, p)) for p in data["positions"]]
```
