# Final 保底包 — GPU 機 Linux 驗證 runbook（可直接照抄）

> ## ✅ 2026-08-07：這關已經跑完，兩輪 ALL PASS
> `op_wrapper.py` md5 `445118482de5f128a23ffc48583691a4` 那顆包，在 GPU 機 WSL2 上
> round 2（預設 `1.293461035226291`）與 round 2b（`final48` `1.2666234250706565`）
> **都通過** ⇒ Win/WSL 雙邊逐位相同，**M80 的 48 核路徑已有 Linux 上的硬證明**。
> **下面的步驟只有在「包又換過」時才需要重跑**——屆時 §0 的三個 md5 與 §3 的兩個
> 預期值都要跟著換，尤其 `ANCHOR48`（`m67c_make_linux_bundle.py` 內嵌的 `_TIER3_PY`）。

> 這台開發機**沒有 WSL / Docker / Linux bash**（2026-08-03 首測、2026-08-07 複測，`wsl -l -v`
> 只印 usage、無任何 distro）⇒ 這一關只能在 **GPU 機的 WSL2 Ubuntu-22.04** 上跑。
> 跑完 Final 保底包就零風險了。
>
> ⚠️ **這是給 Final（2026-08-21）的包，不要拿去覆蓋 Beta。** Beta 已上傳 M73
> （`op_wrapper.py` md5 `c2e27c99…`），使用者 08-01 已裁示不換件。
>
> 🆕 **2026-08-07 重打包：包內容從 M74 換成 M80**（cores-gated knob-cloud tier）。
> 下面的三個 md5 與 round 2b 的預期值**全部是新的**，舊版數字一律作廢。

## 0. 身分確認（在這台 Windows 上先對一次，貼給 GPU 機時才知道有沒有帶錯檔）

| 物件 | 路徑 | md5 | 大小 |
|---|---|---|---|
| Linux 驗證 bundle | `C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz` | `06c8a4710496968fb5c2f823dff1a4de` | 161,613,096 B |
| 送件 tar | `build_submission\cadc1075.tar.gz` | `1ffd503d4ae18933733cd342362ec5ab` | 309,696 B |
| **entry（唯一穩定身分）** | `build_submission\cadc1075\op_wrapper.py` | **`445118482de5f128a23ffc48583691a4`** | 110,089 B |

2026-08-07 建包當下雜湊。**tar 的 md5 不可重現**（gzip 內嵌 mtime；同一批檔案重 stage
就會換一顆，本輪實測 309702 → 309696 B），所以身分一律看 `op_wrapper.py`。
建 bundle 那步會重跑 `make_submission.stage()` ⇒ 上表的 tar md5 是**建 bundle 之後**重新量的，
並已逐檔確認 tar 內 6 個 member 與通過 verify 的 stage 目錄 md5 完全相同。

```powershell
Get-FileHash "C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz" -Algorithm MD5
Get-FileHash "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\build_submission\cadc1075\op_wrapper.py" -Algorithm MD5
```

## 1. 把兩個檔搬到 GPU 機

`m67c-linux-verify.tar.gz`（162 MB，內含資料集、原始碼、腳本、兩顆錨檔、**以及一顆內嵌的
cadc1075.tar.gz**）與 `build_submission\cadc1075.tar.gz`（要送的那顆）。

⚠️ 內嵌那顆與磁碟上那顆是**同一個 md5**（`1ffd503d…`，建 bundle 時由 builder 自己印出來比對過），
所以 round 2 驗的就是要送的東西。

## 2. 在 WSL2 裡解開並備妥環境（只有第一次要跑 setup_env）

```bash
mkdir -p ~/final80 && cd ~/final80
tar xzf /mnt/c/Users/Nordra/Downloads/m67c-linux-verify.tar.gz
cd m67c
bash setup_env.sh
```

`setup_env.sh` 是冪等的：裝 g++、建 `~/m67c_venv`，並確認
`torch / numpy / shapely / tqdm / requests / matplotlib` 都在。末行應印 `setup_env: OK`。

## 3. 唯一還沒做的那一關：round 2

```bash
cd ~/final80/m67c && bash verify_final_tar.sh /mnt/c/Users/Nordra/Downloads/ICCAD2026_FloorSet/FloorSet/build_submission/cadc1075.tar.gz
```

腳本會跑**兩輪**（`m67c_tier3.py final` 與 `final48`）：

| 輪次 | 組態 | 預期逐位總分 | 錨檔 |
|---|---|---|---|
| round 2 | 預設（WSL `nproc`=16 ⇒ 高核 tier 全部惰性） | **`1.293461035226291`** | `results_M80_default.json` |
| round 2b | 強制 `ICCAD_ADAPTIVE_CORES=48` ⇒ **tier-5 與 M80 tier 同時觸發** | **`1.2666234250706565`** | `results_M80_c48_on.json` |

兩輪都必須 **100/100 feasible**、`|d|=0`（允許 <2e-9 的 ULP warn 帶）、
且**不可出現 `constructive.exe` 編譯產物**（那是 bundled-binary-first 的硬證明）。

**成功的判準只有一行**：

```
VERIFY_FINAL_TAR: ALL PASS
```

### 🆕 round 2b 這輪這次特別重要

M74 時代兩輪的預期值**是同一個數字**（48 核只是把 tier-5 打開，總分不變），所以 round 2b
形同 round 2 的副本。**M80 起不是了**：

- 48 核那輪與預設輪**有 58 案版圖不同**，總分差 −2.075%；
- `_M80_CORES_MIN` 與 `_M67F_CORES_MIN` 同為 **40** ⇒ 這一輪是 tier-5 與 M80 tier **一起**翻開；
- **這是 M80 那 8 隻新 profile 唯一會被執行到的地方**，也是它們第一次在 Linux 上跑。

⇒ ULP warn 若不只出現在 case 84 屬**可預期**（58 案是新版圖），超過 `2e-9` 才算 FAIL。
若 round 2b FAIL 而 round 2 PASS，**問題在平台不在打包**：2026-08-07 已在 Windows 上用
**解開後的套件** + `ICCAD_ADAPTIVE_CORES=48` 跑過，逐位得到 `1.2666234250706565`、
0 個 ULP warn、100/100 feasible。

## 4. 常見坑

- **看不到 `final48` 字樣** ⇒ 你解開的是舊 bundle。`grep -l final48 *.py` 應該命中
  `m67c_tier3.py`；沒命中就是 md5 帶錯了，整包重傳（bundle 不能只換單檔）。
- **錨檔找不到**（`results_M80_*.json`）⇒ 一樣是舊 bundle：這兩顆是 M80 才換進 `_SOURCES` 的。
- **`FATAL: venv missing`** ⇒ 沒跑 `setup_env.sh`，或 `~/m67c_venv` 被清掉。
- round 2b 的那行 `WSL nproc=16 < 40 -> tier-5 stays OFF by default` 是**說明文字不是錯誤**，
  它接著就會強制打開高核分支。
- 兩輪之間不要中斷；`rc` 是累積的，只有全過才印 ALL PASS。

## 5. 過了之後

1. 把 `build_submission/cadc1075.tar.gz` 覆蓋到 Google Drive 的 **Final** 位置。
2. 回報 `op_wrapper.py` 的 md5（應為 `44511848…`）當作上傳身分紀錄。
3. 在 `CLAUDE.md` 的「📦 送件狀態」把「還沒做」那兩項劃掉。
