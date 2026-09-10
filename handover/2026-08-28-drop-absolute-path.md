# 給組員 — 拿掉包裡最後一條絕對路徑

**patch：`handover/2026-08-28-drop-absolute-path.patch`**（2,640 B，兩個檔）

---

## 0. 🚨 先講一件跟 patch 無關但更重要的事

**Final deadline 是 2026-08-31 23:59:59，不是 08-28 17:00。**
主辦另外寄了信給參賽者。你樹上每一份文件（`HANDOFF_2026-08-2*.md`、
`VERIFY_RUNBOOK_2026-08-27.md`）都寫 08-28 17:00 ⇒ 你以為今天就截止了。
**還有三天。**

---

## 1. patch 做什麼

`op_wrapper.py:1709` 的

```python
if os.name == "nt":
    compilers.insert(0, r"C:\msys64\ucrt64\bin\g++.exe")
```

拿掉，順手把 `make_submission.py` 的 `_ABS_ALLOW` 白名單一起刪掉，
讓 `_hygiene` 對**任何**絕對路徑都是硬錯。

改的是 **source**（`optimizer_constructive.py` + `make_submission.py`），
不是 staged 檔；重新 `stage` 就會帶進去。

## 2. 為什麼值得改

- `l246_compliance.py` 那 20 條裡**唯一的 FAIL 就是它**（A 52-53 / 112-113
  「no absolute paths in code」）。我方用一把獨立寫的尺重掃，得到**同一條**，
  行號也相同 ⇒ 不是你的 checker 太嚴。
- 你的辯護（在 `os.name=="nt"` guard 內、POSIX 到不了、M73 帶著它被實際評分過）
  **完全成立**——但這是在賭審查方看的是「可達性」。**checklist 掃的是文字。**
- 移除在評分機上**成本恰好為零**：POSIX 走 bundled ELF、根本不編譯；就算走到
  fallback 編譯鏈，guard 也排除它。本機也已多餘——裸 `g++` 解到同一顆 msys
  binary，`ICCAD_CXX` 可指定不在 PATH 上的編譯器，`l113_ship_gate` 的
  `_cxx_preflight` 在沒人回應 `--version` 時會自己補 msys bin 目錄。
- 我方 2026-08-23 已在自己那條線上做過同一件事（commit `aaa4069`），
  官方 evaluator @48c **cost 100/100 且 positions 100/100 逐位不變**。

## 3. ⚠️ 一個我方踩過的坑

第一版的註解**把那條路徑原文寫進去解釋自己**，結果 `stage` 直接拒絕：

```
HYGIENE FAIL: absolute path in op_wrapper.py:1127: # A hardcoded C:\msys...
```

**grep 型的審查者不管字串在註解裡還是在引數列，閘也不該管。**
所以 patch 裡的註解是「描述」那條路徑，不是「引用」它。
一般化的教訓：**一條「檔案裡不准出現什麼」的規則，不會因為你把那一行變成
沒有作用就被滿足。**

## 4. 套用前我方已經驗過的事

用你 `make_submission.py` 裡那條 `_ABS_RE` **逐字**（先在已知答案上自檢過
regex 本身），掃 D 的五個出貨文字檔：

```
op_wrapper.py      1 hit   1709: compilers.insert(0, r"C:\msys64\...\g++.exe")
op_src.py          1 hit   （同一行，op_src 是 op_wrapper 的逐位副本）
constructive.cpp   0 hit
README.md          0 hit
requirements.txt   0 hit
```

⇒ **拿掉 `_ABS_ALLOW` 之後不會誤傷任何其他東西**，`stage` 的 hygiene 會直接過。

patch 本身也自驗過：在暫存 git repo 上（基底 = 你 `final-2026-08-26-verified`
的那兩個檔）`git apply --check` 與 `git apply` 都 OK，套完的內容與預期**逐位相同**。

## 5. 怎麼套

```bash
git checkout final-2026-08-26-verified
git apply --check handover/2026-08-28-drop-absolute-path.patch   # 先確認
git apply         handover/2026-08-28-drop-absolute-path.patch
```

套完之後：

1. `python make_submission.py stage` — hygiene 應該過（若不過，看第 3 節）
2. **in-set 逐位閘**：`ICCAD_ADAPTIVE_CORES=48` 跑官方 eval，
   對 `results_L237_post.json` 比 **cost 與 positions 都要 100/100**。
   這個改動不碰任何求解路徑，任何一案動了都代表套錯地方。
3. `python l246_compliance.py <新 tar>` — **應該變成 20/20**
4. Linux 五車道（`l238_wsl_final.sh`）照跑

**`constructive.cpp` 與 `bin/constructive_linux` 完全不動 ⇒ 不需要 Linux 重建。**

## 6. 要不要為了它單獨重傳？

**不要。** 這件事的期望值太小，撐不起一次獨立的重傳 + 重驗鏈（約 1.5 小時）
再加上「人為傳錯目錄」這個本 ledger 已經記錄過一次的風險。

**做法：併進星期日那顆包。** 反正那顆要重 stage、重驗、重傳，
這個 patch 的邊際成本是零。

⚠️ 套了之後 **`op_wrapper.py` 的 md5 會變**（不再是
`1c326784de7cd9246cd1f380e2842668`），`VERIFY_RUNBOOK` 的身分欄要一起更新，
而且**交件時要把新 md5 給我方**——我方的驗收閘是拿它當硬比對的。

## 7. 我方驗收方向另外要的兩樣東西

跟這個 patch 無關，但一起講完：

1. **`beta_2026-08-16/beta_evaluation_results.json` 與整個 `beta_2026-08-23/`
   不在 git**，`l146_rf_price.py:55` 還硬編 `C:/Users/.01/Downloads/`。
   缺這些，我方**驗得了品質、驗不了名次**——你 08-24~08-26 那一整串
   RF 定價我方一條都重算不了。

2. **`runtime_factor` 的分母是「跨隊中位數」、每一輪重算**
   （`iccad2026_evaluate.py:552`），Final 的 median ≠ beta 的 median。
   **beta 內部就已經漂過一次**：08-16 與 08-23 兩份排行榜之間，我們的
   `raw = 1.320665` 與 `runtime = 52.07s` 完全沒變，`total` 卻從
   `0.924518367` 變成 `0.92659` ⇒ cwRF **0.70004 → 0.70159**。

   而 `_M49_REFINE_BAND` 2/2、`_L196_LPGATE` 的
   `t ≤ 0.3046·M(n)·1.2`、L211 pool drop **全部錨在 beta median 上**，
   沒有任何一份文件處理漂移。想要一張 **median 全域縮放 m ∈ [0.8, 1.3]
   的敏感度表**，三個 arm 對照（現況 / REFINE 還原 / LP 閘開滿 100）。

   方向性是重點：Final median **變大**（頂端隊伍都在用時間換品質，
   rank 1/2 都跑 110s）⇒ 預算變大、我們本來就坐在 floor 上 ⇒
   為了 RF 付掉的那 **2.69% 品質**（L165 的 1.19432 → D 的 1.22641）
   **是白付的**。
