# Final 保底包 — GPU 機 Linux 驗證 runbook（可直接照抄）

> 這台開發機**沒有 WSL / Docker / Linux bash**（2026-08-03 實測 `wsl -l -v` 無任何 distro、
> `docker`/`bash` 皆 not found）⇒ 這一關只能在 **GPU 機的 WSL2 Ubuntu-22.04** 上跑。
> 跑完 Final 保底包就零風險了，與 M78 之類的品質軸完全獨立。
>
> ⚠️ **這是給 Final（2026-08-21）的包，不要拿去覆蓋 Beta。** Beta 已上傳 M73
> （`op_wrapper.py` md5 `c2e27c99…`），使用者 08-01 已裁示不換件。

## 0. 身分確認（在這台 Windows 上先對一次，貼給 GPU 機時才知道有沒有帶錯檔）

| 物件 | 路徑 | md5 | 大小 |
|---|---|---|---|
| Linux 驗證 bundle | `C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz` | `b3989af0a0958488a77df6629cef6d04` | 161,616,892 B |
| 送件 tar | `build_submission\cadc1075.tar.gz` | `d529c7828e2e8a36a7165e70d9a22ee0` | 305,709 B |
| **entry（唯一穩定身分）** | `build_submission\cadc1075\op_wrapper.py` | **`ce4f34716ea14863e62f68d6970e983d`** | 102,751 B |

2026-08-03 重新雜湊，三者與 ledger 逐位相符、未漂移。
**tar 的 md5 不可重現**（gzip 內嵌 mtime），所以身分一律看 `op_wrapper.py`。

```powershell
Get-FileHash "C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz" -Algorithm MD5
Get-FileHash "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet\build_submission\cadc1075\op_wrapper.py" -Algorithm MD5
```

## 1. 把兩個檔搬到 GPU 機

`m67c-linux-verify.tar.gz`（161 MB，內含資料集、原始碼、腳本、**以及一顆內嵌的
cadc1075.tar.gz**）與 `build_submission\cadc1075.tar.gz`（要送的那顆）。

⚠️ 內嵌那顆與磁碟上那顆是**同一個 md5**（`d529c782…`，08-01 建 bundle 時確認過），
所以 round 2 驗的就是要送的東西。

## 2. 在 WSL2 裡解開並備妥環境（只有第一次要跑 setup_env）

```bash
mkdir -p ~/m78final && cd ~/m78final
tar xzf /mnt/c/Users/Nordra/Downloads/m67c-linux-verify.tar.gz
cd m67c
bash setup_env.sh
```

`setup_env.sh` 是冪等的：裝 g++、建 `~/m67c_venv`，並確認
`torch / numpy / shapely / tqdm / requests / matplotlib` 都在。末行應印 `setup_env: OK`。

## 3. 唯一還沒做的那一關：round 2

```bash
cd ~/m78final/m67c && bash verify_final_tar.sh /mnt/c/Users/Nordra/Downloads/ICCAD2026_FloorSet/FloorSet/build_submission/cadc1075.tar.gz
```

腳本會跑**兩輪**（`m67c_tier3.py final` 與 `final48`）：

| 輪次 | 組態 | 預期逐位總分 | 錨檔 |
|---|---|---|---|
| round 2 | 預設（WSL `nproc`=16 ⇒ tier-5 不觸發） | **`1.293461035226291`** | `results_M74_default.json` |
| round 2b | 強制 `ICCAD_ADAPTIVE_CORES=48` ⇒ **tier-5 真的跑到** | **`1.293461035226291`** | `results_M74_cores48.json` |

兩輪都必須 **100/100 feasible**、`|d|=0`（M74 允許 case 84 落在 <2e-9 的 ULP warn 帶）、
且**不可出現 `constructive.exe` 編譯產物**（那是 bundled-binary-first 的硬證明）。

**成功的判準只有一行**：

```
VERIFY_FINAL_TAR: ALL PASS
```

## 4. 常見坑

- **看不到 `final48` 字樣** ⇒ 你解開的是舊 bundle。`grep -l final48 *.py` 應該命中
  `m67c_tier3.py`；沒命中就是 md5 帶錯了，整包重傳（bundle 不能只換單檔）。
- **`FATAL: venv missing`** ⇒ 沒跑 `setup_env.sh`，或 `~/m67c_venv` 被清掉。
- round 2b 的那行 `WSL nproc=16 < 40 -> tier-5 stays OFF by default` 是**說明文字不是錯誤**，
  它接著就會強制打開高核分支。
- 兩輪之間不要中斷；`rc` 是累積的，只有全過才印 ALL PASS。

## 5. 過了之後

1. 把 `build_submission/cadc1075.tar.gz` 覆蓋到 Google Drive 的 **Final** 位置。
2. 回報 `op_wrapper.py` 的 md5（應為 `ce4f3471…`）當作上傳身分紀錄。
3. 在 `CLAUDE.md` 的「📦 送件狀態」把「還沒做」那兩項劃掉。
