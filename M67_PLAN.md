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

### M67-C:Linux static binary + 四關驗證 ✅ 完成(2026-07-21)
1. 新工具 `m67c_make_linux_bundle.py`(永不 ship):產自足 WSL2 驗證包 `Downloads/m67c-linux-verify.tar.gz`(161MB / 422 檔;內嵌 scripts 強制 LF、現產 staged `cadc1075.tar.gz`、8 隻 loader 閉包、`LiteTensorDataTest`、`results_shipped_m51.json`)。包內 `verify_final_tar.sh` 供最終 tar 複驗(免重傳大包)。
2. **WSL2 Ubuntu-22.04 / g++ 11.4.0 / python 3.10.12 / torch 2.11.0+cpu 四關 ALL GREEN**:
   - **T1** `-O3 -std=c++17 -static-libstdc++ -static-libgcc`(無 -march,保 portable)→ `constructive_linux` 508248B、`sa_linux` 350352B,ldd 僅剩 libc/libm;全 `-static` 備胎亦 COMPILE_OK(1485672B,**不入包**、留 Final 保險);三隻 1-block SMOKE OK。md5:cons `70d3cd9f…a301`、sa `e89d3a84…1325`。
   - **T2** `m48_coldstart_dryrun.py opwrapper` 四 phase ALL PASS(phase 4 僅剩 nt-gated msys 行)。⚠️ gotcha:phase 2 的假 binary 以 `MZ` 開頭 → 撞 WSL binfmt `WSLInterop` → Windows 彈「不支援的 16 位元應用程式」modal(**無害**,子行程非零 → smoke 正確拒收 → 重編 PASS;免對話框:跑前 `sudo sh -c 'echo 0 > /proc/sys/fs/binfmt_misc/WSLInterop'`)。
   - **T3** 官方指令 100 案:total **1.326473104916827 逐位 |d|=0.000e+00**、100/100 feasible、avg **0.92s**(Windows 1.73s)、eval 102s;唯一 ULP-WARN = case 84 cost |d|=**4.441e-16**(positions 逐位相同,比 2026-07-15 的 <2e-9 更緊);**bundled-first 生效鐵證 = 跑完無 `constructive.exe` 編譯產物**、stderr 零 fallback/compile 行。
   - **T4** 破壞 `bin/constructive_linux`(垃圾 bytes)→ 落編譯鏈:case 50 cost 1.3106535809308177 逐位 = anchor ∧ `constructive.exe` 出現 → POSIX 上 layer-1→layer-2 交接證實。
3. 入包 + gate 重跑(Windows):binaries → repo `bin/`(md5 對照 build manifest 逐位)→ `make_submission.py all` 8 檔 hygiene OK、tar `bin/*` mode 0o755、官方佈局 verify 逐位 **1.326473104916827**;交叉驗證 = 最終 tar 內 `op_wrapper.py` md5 `709a255e…958f` **與 WSL T3 實跑那份逐位相同**;`m48 opwrapper` 四 phase + `regression_suite` 六項重跑存證（752s ALL PASS）。
   **第二回合(最終 tar 端到端)✅**:含 bin/ 的實際上傳物（`cadc1075.tar.gz` 425563B md5 `b9589618d507de0561f79a55a80fd8f3`）帶回 WSL2 跑 `bash verify_final_tar.sh` —— **零注入**、包內自帶 binaries：total **1.326473104916827 |d|=0.000e+00**、100/100 feasible、`bundled-first: OK`、唯一 case 84 4.441e-16、avg 1.15s、eval 128s → **Linux 端定案**。
4. 風險註記:grader glibc 未知 → partial-static 綁 build 機 glibc(2.35)向前相容,風險低;真不合 → 第二層現場編譯(T4 已證)、第三層 SA。fullstatic 備胎存 `Downloads/m67c_bin_out/`。

### M67-D:OOS 泛化預檢 ✅ 完成(2026-07-21)——**送件形不動;RF 投影須改用 OOS 稅**
1. `m67_oos_probe.py`(永不 ship;`gate0|run|report|ref|pool0`,cache `m67_oos_cache.pkl`,dump `results_M67D_oos.json`):訓練集 `floorset_lite` 抽 **240 案**(seed 67;validation 每個 n∈[21,120] 恰 1 案 → 鏡射之,每 n K=2、n>100 K=4),baseline 逐位鏡射 `_extract_baseline`(含 stored-metrics 分支,訓練 metrics_sol 與重算 dev 2.8e-8),評分 = 官方 `evaluate_solution` + `target_positions` + `median_runtime=1.0`(RF=1.0 同 in-set 語意),估計量 = per-n 先平均再官方加權。**gate0 全 PASS**(env 淨、池=41、in-set 3 案 cost+positions 逐位 = `results_shipped_m51.json`、訓練 fp_sol verbatim feasible∧hgap≈0)。
2. **結果**:OOS raw **1.6533** vs in-set 1.3265(+24.6% → 預註冊 bar 判 RED);floor-relative ratio 1.3287 vs 1.1972(+10.98%);**硬旗標全清**(100% feasible、零 fallback、零例外、runtime p50 1.54s vs 1.55s、p90 2.44 vs 2.58)。band:S 0.6%/M 18.2%/**B 81.1% wContr**,B 帶 1.6599 vs 1.3121。
3. **歸因(raw RED ≠ 過擬合)**:(a) 單 profile 參照 in-set 1.4775→OOS 1.8681 = **+26.4% gap 比 portfolio 的 +24.6% 更大**、portfolio 增益 in-set 10.22% vs **OOS 11.50%** → 品質軸調參泛化;(b) 語料本身更硬:label floor 1.2444 vs 1.1079(label vrel 0.1081 vs 0.0504)、boundary blocks 28.8 vs 24.0、preplaced 3.08 vs 2.59、b2b 1201 vs 994。
4. **🚨 主發現(`pool0` 模式)**:M41/M42/M45+M49/M50 的 adaptive 切法品質稅 **in-set +0.106%(movers 3/20)vs OOS +2.825%(movers 52/80,最壞單案 +11.9%)= 27 倍**;證實 M55 CV 預言(且更嚴重:break 65% vs CV 的 40-48%)→「strict selection-preserving ⇒ ∀median∀cores 弱贏」**只在樣本內成立**。**但仍不改送件形**:pool-cut wall 比值 2.50× @48c(實測含 REFINE 為 7.28× @12c),以 alpha 校準(RF=0.7081≈floor ⇒ t≈0.30·median)算 `cost_full/cost_ship = (1/1.02825)×(0.921/0.700) = 1.279` → 切法淨賺 **~28%**(含 REFINE 取 5× 則 ~58%);淨虧只在 median ≥ **8.2×** t_shipped(alpha 實測 3.28×,安全邊際 2.5 倍)。
5. **對 M67-E 的交接**:48c 投影**必須改用 +2.8% 的 OOS 品質稅**取代 in-sample 的 +0.1%。細節見 `M67D_REPORT.md`。

### M67-E:48c RF 投影 ✅ 完成(2026-07-21)——**送件形不動;但發現 M42/M45 在 48c 買不到 RF**
1. `rf_score_model.py` cores 網格加 48(四處)+ 48c 結構表(逐帶 `|P|`/`max_i`/`Σ48`/crossover `c*`);asserts 全未動、全綠。新工具 `m67e_rf48.py`(永不 ship;`gate0|calib|fit|project|report`,cache `m67e_cache.pkl`,dump `results_M67E_rf48.json`)。**gate0 五閘 ALL PASS**(rf_score_model 子行程 exit 0 + 六個 marker / audit sig / alpha JSON 逐位對齊 |d|=2.2e-16 / **48c tier fail-open 100/100 且 @4c 對照 40 池會變** / 48c wall max-bound 100/100)。
2. **🚨 校準修正(改寫 M67-D §3 算術)**:alpha 送的是 **M10 的 14 隻廉價 knob 池**(逐案 grader t p50 **0.673s**、大案 2.4-4.3s),與我們現行 shipped 本機 12c(p50 1.547s、大案 1.8-2.4s)**同量級** → median 必須逐案錨 `M_i = κ·t_i^alpha`(**κ=3.161**;bracket 模型 B 常數 M=9.43s、97/100 clamped)。**M67-D 的「median ≈ 3.28× t_shipped、8.2× 安全邊際」作廢**:實際是**貼著 floor 邊緣**(重帶 6/10 觸底、h p50 1.0-1.3)、**mid band 0/40 觸底**(加權 RF 0.82)。機速 s bracket [1.5, 1.7](兩者皆上界);投影對 s∈{1,1.5,2,2.5} 全掃。
3. **wall 模型驗證**(`fit`):`W=max(max dt, Σdt/cores, ΣPT)`;40 隻 REFINE-free 案 OLS → **a=0.9997、b=2.45ms/profile、c≈0**;γ 逐帶 = 1.20/1.06/0.88/**0.496**/0.463(後兩者正是 M49 K=4 的 −50%);full-pool 側用 M67-D pool0 實測 → γ_full 1.088、**實測 full/ship wall 6.27× @12c**。M47 後 proxy 鏈 ≈2.5ms/隻(ΣPT ≤0.09s,ledger 的 2.9s 是 pre-M47)。
4. **投影結論(κ=3.16;κ∈{2.5,4,6}、模型 B、`--tax-all`、`--cores 24` 符號全同)**:shipped @s=1 **0.9926**(加權 RF 0.7325)、s=2 1.1668;**`POOL=0` 全池 +52.8~+60.0%**、**REFINE 還原 K=12 +15.1~+20.5%** → **M41 swap 砍 + M49/M50 REFINE 是 48c 的 RF 主力,必須留**。
5. **🚨 主發現**:48c wall=max-setter ⇒ 被 M42/M45 砍掉的 22 隻(全部 dt ≤ max-setter)**加回來 dW = +0.00%**(重帶 13→33/34 隻、ΔΣPT ≤0.20s、`c*(restore)` ≤27.2 → **連 24 physical 都 ≤+0.53%**);in-set dQ ≈ 0 是設計使然(閘就是逐案 cost 相等)。⇒ **M42/M45 是為 12 核 `Σ/cores` 牆設計的,Beta 的 48 核把那面牆拿掉了,只剩 OOS 品質稅**:break-even **θ\*=0.000**(回收任何一點就贏)、θ=1 上界 **−2.11%**。
6. **交接 M67-F(未做,送件形不動)**:θ 只能實測——`m67_oos_probe.py` 加「高核 restore 池」env 形(留 M41+M49/M50、跳過 `_BIG_REDUNDANT_IDX`+`_M45_BAND_DROP`)在同批 80 OOS 重案重跑 `pool0`;θ 顯著 >0 才討論 ship,且須做成 **cores-gated tier-5**(偵測失敗→現行行為 fail-safe)並重跑 m49 三 gate → regression_suite → make_submission → WSL 逐位複驗。⚠️ deadline 2026-07-31,現行 tar 已四關綠,θ 未量到前不要動 wrapper。細節見 `M67E_REPORT.md`。

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
