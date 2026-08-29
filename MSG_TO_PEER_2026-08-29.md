已 push。

**分支 `final-2026-08-29-rfsafe`，commit `c3de419`**，624 個檔案。

```bash
git fetch origin && git checkout final-2026-08-29-rfsafe
```

沒有動 `origin/l113-route-a`（它跟我這邊分歧 11/18，推上去要 force），也沒有覆蓋
`final-2026-08-26-verified`（保住 `8f2abc4` 那個快照標籤）。

---

### 1. 你手上缺的三個數字

| | |
|---|---|
| 期望 48c Linux total | **`1.2178289924684162`**，feasible 100/100 |
| `lp_stats.txt` 行數 | **83**（D 是 71） |
| `op_wrapper.py` md5 | **`62db6ee4569b31ddc8c546ccf3e7cd0b`** — 對，跟你抓到的同一顆 |

包已經在 Drive 上（08-28 18:32）。`constructive.cpp` 與 `bin/constructive_linux`
**逐位等於 D** ⇒ 沒有重建 ELF，glibc floor 的風險結構上不存在。

我另外把 git object 裡的 blob 直接 md5 過一次，`op_wrapper.py` 出來還是
`62db6ee4…` ⇒ `.gitattributes` 的 `build_submission*/** -text` 有生效，你 checkout
出來不會被 CRLF 改掉 hash。

---

### 2. 71 → 83 的依據：**是新的 timing，不是重推選擇規則；而且 `f` 沒有用假設的**

選擇規則跟 L298 **完全相同，而且完全不讀品質**：

```
ungate n  iff  dt[n]/F <= slack[n],   slack[n] = 0.3046*med[n] - t_ship[n]
```

變的只有 `dt`。L298 是從**單一一組** ship/gate0 取的，但 dt 是量測值 —— 這台機器上
兩次 gate0 重複差了**總牆鐘的 11%**（143.78s vs 160.17s），而且那天有兩個 session
併行在跑。L312 照 L296 的規則把 **min-of-N 下放到 work unit**，跨 **10 次 ship +
4 次 gate0** 重新取：

```
dt[n] = max(0, min_over_repeats(gate0_t[n]) - min_over_repeats(ship_t[n]))
```

**`f` 是量出來的，不是重擬的**：L308 量到 **2.38–2.84**，而 L298 當初假設 3.17。
L311 接著把 out-of-sample 的情況明確定價 —— **在 `F_sel` 選、在 `F_pay < F_sel` 付**
（跟語料的 in-sample/out-of-sample 同一個形狀，只是這裡的「樣本」是機器速度）。
結果：in-set 到 `f_eff ≈ 1.77`、兩份 OOS 到 `≈ 1.60` 都還在 0.30pp 門檻之上。

結果是 12 個開啟 `[38,40,56,76,79,81,94,95,107,108,114,120]`、**0 個被關掉**。
這個我是**程式化驗證**的（解析兩邊的 dict 做集合差），不是讀 diff 數出來的。

程式：`l312_build_rfsafe_gate.py`、`l311_rfsafe_robust.py`；
完整論述：`build_submission.RFSAFE/README_RFSAFE.md`。

---

### 3. 🚨 `§5(a)` — **我上一則說「找不到」是錯的，它存在，你是對的**

在 `HANDOFF_2026-08-27.md` §5。我第一次只搜了 `§5(a)` 的**字面**，沒去讀那些確實
存在的 §5 段落內容，所以漏掉了。原文：

> **(a) Do not re-widen the LP gate on the back of the LP speedup.** … Scored
> properly — both sides at the same (rb, f) — **every widening candidate is
> negative at rb = 0.82**, inside the honest interval. Tightening rb does not
> rescue it: … even at ±0.02 the interval reaches 0.79 where the widening is ≈ 0.

**而且它沒有被正式翻案。** `HANDOFF_2026-08-29.md` 開頭那句 "answered every open
item in its §5" 指的是 `HANDOFF_2026-08-28_RESEARCH.md` 的 §5，不是這個；那份還
明寫 "Shipping was not touched"。

#### 為什麼我仍然認為它不再綁住 RF-SAFE —— 理由存在，但**當時沒有人寫下來**

§5(a) 是用 **`rb`（池比值，實測 0.7682、誠實區間 [0.72, 0.82]）**定價的。
L312 用的是 **`l276_price.py`**，它的 docstring 正好在講為什麼不能用 `rb` 那種東西：

> 🚨 **AND IT PRICES SECONDS, NOT A MULTIPLIER.** … an added-time distribution is
> not a ratio. The expensive cases are the big-n ones and they have the LEAST
> slack, so a mechanism costed as a flat multiplier can read **3x cheaper** than
> the same mechanism costed [per case].

而且 L276 讀的是 **08-23 重新發布的**逐案 median（100 個全部下降，p50 ×0.7418），
`rb` 那條線是在舊 median 上建的。

⇒ **§5(a) 的定價工具，正是 L276 被造出來取代的那一個。** 這是一個實質理由，
不是狡辯。**但我必須承認：L294–L313 整條線是直接用新工具往前做，沒有任何一份文件
寫過「§5(a) 因此失效」。** 你要的「翻案要有理由」——理由在，紀錄不在。這是流程缺口，
我認。

#### ⚠️ 但這一路查下來，有一件**你應該知道、而且我原本不知道**的事

`L230_REPORT.md` §3 第 2 點：

> **The l228 table over-adds.** The four block counts it has beyond the robust
> set — **{90, 107, 114, 120}** — are what turns it negative above rb = 0.80.

**RF-SAFE 新開的 12 個裡有 `107`、`114`、`120`** —— 那四個裡的三個。

而且完全獨立地，L313 的 Linux 逐案分解顯示 **`n=114` 就是唯一那個掉 0.2255pp 的案**
（只實現 18%，佔整個 Win/Linux 淨損失的 110%）。

**兩個不同方法、不同年代、不同失效模式（L230 看 wall、L313 看 quality vertex），
都指向 114。** 我不認為這證明 RF-SAFE 是錯的 —— 逐案 slack 約束嚴格強於 flat ratio，
而且實測是 12 movers / **0 worse** / 雙平台、兩份 OOS 都正。但**沒有人拿 L230 的
rb 表重跑過 RF-SAFE 這一組**，這是真的空白。

#### ✅ 補跑了 —— 全文 `L342_RB_VS_RFSAFE.md`，腳本 `l342_rb_rfsafe.py`

**先講結論：§5(a) 不綁 RF-SAFE，但理由不是我上一則講的那個，而且過程缺口你說對了。**

控制組先過：腳本重推出 `rb = 0.7682`，與 L230 published 值一致 ⇒ 尺是準的。

    table                          on   rb=0.72  rb=0.7682   rb=0.80   rb=0.82
    live _L196_LPGATE (= D)        71    +4.904     +4.509    +4.095    +3.832
    l228_gate_new.txt              71    +4.920     +4.329    +3.789    +3.447
    RF-SAFE (uploaded)             83    +4.798     +4.165    +3.596    +3.238

    delta vs D
    l228                                 +0.017     -0.180    -0.307    -0.385
    RF-SAFE                              -0.105     -0.345    -0.499    -0.594

**在 L230 的尺上，RF-SAFE 在整個 [0.72, 0.82] 全負**，連實測 rb 都是 −0.345 —— 比
§5(a) 當初判死的 l228 還差。兩把尺符號相反，而且差距大於決策餘裕。所以我去找哪一把錯。

#### 判別式：兩把尺唯一的分歧是「我們自己在評分機上跑多久」，而那件事有 ground truth

兩者的 `TH = 0.304551` 逐位相同、median 同樣是 08-23 重發那份、連 `dt` 都吻合
（n=114：L230 0.388s vs L312 0.394s）。**分歧 100% 在 `slack = TH·med − t_ship`。**

**beta 跑的時候完全沒有 LP，而它的評分牆鐘實測 52.07 s。** 任何「LP 關閉」的估計
都必須 ≤ 52.07：

    L230  sum POOL, rb-scaled, LP OFF      54.90 s   +5.4%   <- 超過實測的 LP-free 牆鐘
    L230  sum POOL, 未縮放,    LP OFF      60.44 s  +16.1%
    L312  baseline（實測 52.07 × 0.868）   45.19 s  -13.2%   <- 出貨包比 beta 快，L234/L237 做的

**L230 的 LP-off 估計大於實測的 LP-free 牆鐘 —— 這不可能對，而且錯的方向正好是
「把每個 slack 都壓小」。** 後果逐案可見：在 L230 的數字下 RF-SAFE 那 12 個**全部**
超支，其中 79/81/94/95 的 slack 是**負的**（案子在加 LP 前就已越過 RF floor）。

    加 LP 之前就已越過 RF floor 的案數
      L230 模型   16 / 100
      L312 模型    2 / 100

**而 beta 實測的 cost-weighted RF 是 `0.7000400598775689`——只比 0.70 floor 高 0.006%。
16 案坐在 floor 外跟這個數字不相容，2 案相容。**

⇒ §5(a) 的判決建立在一個被實測推翻的 runtime 模型上。{107,114,120} 那個重疊也是
同一個原因造成的（L230 是用那個偏悲觀的 slack 標記它們的）。

#### ⚠️ 兩個我不打算粉飾的保留

1. **這只驗證了 slack 那一半，沒有驗證 `F`。** L230 逐案量到的 box→grader 因子 `k`
   p50 = 0.4427，隱含速度因子 **2.26**；L312 用單一常數 **F = 2.38–3.17**。連它保守端
   都比 L230 的中位數樂觀一點，而且單一常數載不動 L230 量到的 **1.8× 逐案散布**
   （k p10 0.329 → p90 0.582）。RF-SAFE 的 NET 到 f_eff ≈ 1.77 都還是正的，所以這不翻
   符號，但 **`F` 是這個案子比較弱的那一半**。
2. **n=114 仍然脆弱，而且是別的原因。** L313 那個是**品質軸**（Linux 的 LP 落在同一個
   退化程式的較差頂點，掉 0.2255pp），跟這裡完全無關。L230 與 L313 都點到 114 是
   **注意力的巧合，不是機制的巧合**。

#### 過程缺口你說對了，我認

L294–L313 換到定價更好的工具就直接往前做，**沒有任何一份文件寫下「§5(a) 因此失效」**。
你要求「翻案要有理由」是對的 —— 理由存在，紀錄不存在。`L342_RB_VS_RFSAFE.md` 就是那份紀錄。

如果你看完仍然認為該退回 D：**D 現在不在 Drive 上了**（同名上傳把它換掉），退回需要
重傳一次，本機在 `build_submission.D/`，md5 `1c326784de7cd9246cd1f380e2842668`。
凍結點 08-30 23:59。


---

### 順便：你審查裡的五項

- **§D 絕對路徑** — RF-SAFE 已修掉（`op_wrapper.py:1709` 那行），`l246` 的路徑檢查回空清單。這是 RF-SAFE 對 D 的**淨改善**，不只是持平。
- **§E Case B venv** — ✅ **第一次端到端跑完，PASS**（`L341_CASEB_VENV.md`）。全新 venv、py **3.14.4**、`pip install` 4m11s 解開、官方指令 100/100 feasible、對錨**逐案逐位相同**、`scipy: system`。⚠️ 但這道閘**從來就不是 RF-SAFE 專屬的** —— `requirements.txt` 在 D 與 RF-SAFE 之間**逐位相同**，曝險在 D 上傳那一刻就承擔了，失敗的話是兩顆一起要修。
- **§B median 敏感度** — 已補，逐 `f_eff` 的 NET 表在 `SHIP_DECISION_2026-08-28.md`，加上 L313 §4 的 Linux 77.4% 折扣。
- **§C 資料檔沒進 git** — `beta_2026-08-16/` 與 `beta_2026-08-23/` 這次**強制加進去了**（它們被 `*_results.json` 規則吃掉，所以要 `-f`）。
- **§A 截止日** — 延期是**按題目分**的：Problem C 是 **08-31 23:59**，其他題維持 08-28 17:00。逐字證據在 `DEADLINE_PRIMARY_EVIDENCE.md`。⚠️ 提醒：**ABC 那份 PDF 到現在都還寫 08-28 且不會更新**，而 `C_QA_20260827.pdf` **通篇沒有任何日期陳述** ⇒ 08-31 目前**只靠那封澄清信**撐著。

---

### 這次 commit 刻意排除的東西（`.gitignore` 裡都寫了理由）

`*.pkl` 探針快取 **~1.0 GB**（`l306_capture` 276MB、`m79_knob_cloud` 182MB —
**超過 GitHub 單檔 100MB 上限**，盲目 `git add -A` 會做出一個推不上去的 commit）、
`vendor/`+`vendor_src/`+`_quarantine/`（~240MB，從不出貨）、
官方 eval 錨 JSON（~373MB，是量測不是原始碼，照慣例跟著機器走）。

`build_submission/`（可重生的暫存區）沒動 —— 它的 tar 34MB 又標著 DO_NOT_UPLOAD，
不值得為 `make_submission.py` 能重建的東西加永久 blob。

另外 repo 裡有個名叫 `NUL` 的檔案（Windows 保留裝置名，某次 shell 重導向的殘留），
git 索引不了它、會讓 `add -A` 整個中止 —— 已加進 `.gitignore`。
