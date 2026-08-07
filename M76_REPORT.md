# M76 — 組員 M73 knob-OFF escape tier，在 M74 底下重量測：**RED**（2026-08-01）

## 判準（開工前寫死，事後未改）

| 項目 | bar |
|---|---|
| in-set 100 加權增益 | 報告用，不單獨判定 |
| **OOS 240 加權增益** | **≥ 0.30%** 才算 quality GREEN |
| 48c wall 代價 | 逐案 `ΔRF=(t_new/t_old)^0.3` 加權，必須 **<** OOS 增益 |
| 可行性 | 100/100 feasible；gate-off 逐位等於 `1.293461035226291` |

## 結論：RED（機制真實、方向正確、但在評分機的池形狀上只剩 1/3）

| arm | 來源集 / 帶別 | in-set 100 | OOS@**16 核**形狀 | OOS@**48 核**形狀 | dRF@48c | **NET@48c** |
|---|---|---|---|---|---|---|
| `m73` | 組員 (2,22,23,25) / 全帶 | +0.243% | +0.302% | **+0.105%** (12 好/1 壞) | +0.088% | +0.017% |
| `m73big` | 組員 (2,22,23,25) / n>100 | +0.192% | +0.288% | **+0.101%** (5 好/0 壞) | +0.020% | +0.081% |
| `m73x` | 我方 (21,23,2,22) / n>100 | +0.319% | +0.294% | **+0.107%** (6 好/0 壞) | +0.020% | **+0.087%** |

**評分機形狀上三個變體全部 ≈ +0.10%，是 bar 的三分之一 ⇒ RED，不出貨。**
送件形零改動：`ICCAD_M73_ESCAPE` 預設 0，gate-off 官方 eval 逐位
`1.293461035226291`、對 `results_M74_default.json` **0 movers**。

### 🔑 主發現：escape tier 與 tier-5 是**替代品**，不是互補品

OOS 240 的 shipped 基準本身隨形狀變動：

| | OOS 240 shipped |
|---|---|
| 16 核形狀（heavy 池 13，M42 砍掉 22 隻） | 1.576749 |
| 48 核形狀（heavy 池 35，**tier-5 把 22 隻放回**） | **1.555855**（−1.325%） |

那 −1.325% 正是 M67-F tier-5 的 OOS 價值（先前在 80 案 held-out 上量到 +2.289%，
加權到 240 案量級吻合）。**tier-5 放回去的 22 隻 knob-ON profile，已經救掉
escape tier 原本要救的大部分案子** ⇒ 同一批分數不能算兩次。

這也直接說明**組員的 +0.288% 是低核數字**：他們在 12 核上量、而且他們的 repo
根本沒有 tier-5，所以他們永遠看不到這個抵銷。

## 1. 方法：knob-ON × knob-OFF 的成對 audit cache

新增 `profile_audit.py esc`（第三個 mode）＝出貨 REFINE overlay + **M71 旗標關**，
產出 `audit_cache_esc.pkl`（42 隻 × 100 案 = 4200 combos，PHASE1 470s）。配上既有的
`audit_cache_ship.pkl`（同 overlay、旗標開），就有了每一個 (case, profile) 的
**knob-ON/knob-OFF 成對位置與 dt**。合併成單一 index 空間（`ESC0+k` = host `k` 的
knob-off 雙胞胎）後，`m76_escape_probe.py` 可**精確**模擬任何來源集的 portfolio。

**保真度：三個端點對真 eval 全部逐位相同。**

| 端點 | 模擬 | 真 eval |
|---|---|---|
| M74 shipped | 1.293461035 | `1.293461035226291` |
| `ICCAD_M71=0` | 1.337769177 | `1.3377691769809372` |
| escape ON（組員集、全帶） | 1.290319514 | `1.2903195142`（11 好 / 0 壞） |

副產物交叉檢查：esc vs ship 有 **3206/4200** 對 (case,profile) 輸出不同、89/100 案
被碰到 ⇒ M71 在 per-profile 層級非常活（也證明 overlay 真的進了 binary；
若差異為 0 則 `profile_audit.py esc` 會直接 abort 而不是產出一顆空 cache）。

## 2. Gate 0：天花板與 48c wall

```
shipped (M74, knob-ON everywhere)  1.293461035
knob-OFF portfolio (same cuts)     1.337769177   (+3.426%)
per-case 2-way ORACLE              1.289063657   (-0.340%)
FULL-UNION escape tier (41 srcs)   1.288443844   (-0.388%)  <- realizable
```

**混合池贏過 2-way oracle**（−0.388% vs −0.340%）⇒ proxy 在聯集上找到兩個端點都
沒有的解。但全 41 隻的 48c ΔRF 是 **+57.0%** ⇒ 天花板買不起。

可回收量 17 案、加權 0.004397 分數點，集中在重帶：(110,inf] 54.7%、(100,110] 28.2%、
(60,100] 16.9%、(0,60] 0.3%。前 5 名 98/83/92/94/82 佔 **82.9%**
（組員量到 98/79/83/92/94，幾乎同一批）。

### 48c wall：組員未解的第 3 項 ＝ 否定

他們擔心「knob-off 逃生口在 48 核會變成 max-setter」。實測 per-profile
`dt_esc / dt_on`：

| band | #2 | #22 | #23 | #25 |
|---|---|---|---|---|
| (60,100] p50 | 0.987 | 1.061 | 1.023 | 1.036 |
| (100,110] p50 | 0.905 | 1.022 | 0.929 | 0.976 |
| **(110,inf] p50** | **0.934** | **0.911** | **0.906** | **0.915** |

重帶的 knob-off 副本比它的 knob-on 本尊**更快** ⇒ 不可能頂掉 max-setter。

| band（組員集、全帶） | 48c dRF | 12c dRF |
|---|---|---|
| (60,100] | +0.064% | +1.150% |
| (100,110] | +0.020% | +0.857% |
| (110,inf] | **+0.000%** | +2.470% |
| **加權** | **+0.088%** | **+4.504%** |

**組員判 all-bands RED（mid wall +3.8~9%）是 12 核產物**——和 M74 把 tier-3 從
universal 降級成 cores-gated 的誤判形狀完全相同。但**「mid 帶不要開」的結論本身
仍成立，理由不同**：實測 mid 帶 OOS 品質只多 +0.014pp（16 核形狀），wall 多
+0.068pp ⇒ 淨 −0.054pp。

## 3. 來源集在 M74 底下重推（48c 池、in-sample）

`m76_escape_probe.py derive`，前向貪婪，NET = 品質 − 48c ΔRF：

| band | 我方 greedy | 品質 | dRF@48c | NET | 組員集同帶 NET |
|---|---|---|---|---|---|
| (60,100] | {24} | +0.037% | +0.008% | +0.029% | **−0.013%** |
| (100,110] | {23,22} | +0.096% | +0.020% | +0.076% | +0.076% |
| (110,inf] | {21,2,23} | +0.223% | **+0.000%** | +0.223% | +0.096% |
| **(100,inf] 合併** | **{21,23,2,22}** | **+0.318%** | +0.020% | **+0.298%** | +0.171% |

`#21` = `CLUSTER_ASPECT=3.0`（M33 破 case 89 的那隻）的 knob-off 副本，組員沒選到。

## 4. 🔑 教訓一：in-sample 優勢的轉移率 ≈ 5%

`m73x` 與 `m73big` 用**同一個帶別 gate、同樣 4 隻**，只差來源集身分：

| | in-set 100 | OOS@16 核 | OOS@48 核 |
|---|---|---|---|
| `m73x` − `m73big` | **+0.127pp** | **+0.006pp** | **+0.006pp** |

我方在 20 個重帶案上貪婪擬合出的額外增益，**95% 沒有轉移**。這是 M72（in-sample
打平藏住 1.4% OOS 差距）、M74（strict 等價 ≠ OOS 等價）、M75（符號翻轉）之後第四次，
而且是**同一機制、同一 gate、只換來源集**的乾淨對照。

⇒ 來源集不值得用 in-sample 貪婪挑；組員「對退步案逐 profile 找救援者」的挑法與
我方加權貪婪 OOS 上等價，那就挑 wall 較低、movers 較少的那個。

## 5. 🔑 教訓二：**OOS 也要挑對池形狀**

in-set 100 案在 16 核與 48 核形狀下給出**完全相同**的數字
（baseline `1.293461`、m73x `1.289345`、5 movers，兩種形狀逐位相同）。
只看 in-set 會得出「形狀無關」的結論——**而 OOS 上形狀值 2.7 倍**
（+0.294% → +0.107%）。

⇒ 「in-sample 藏訊號」這條 doctrine 要再加一層：**它也會藏住組態維度的敏感性**。
任何與 adaptive tier 有交互作用的新機制，OOS 必須在**評分機的核數形狀**下量，
`--force-cores 48` 就是為此而加。

## 6. 誠實範圍

- 48c wall 是**投影不是實測**：用 `audit_cache_ship/esc` 的 dt + M67-E 的
  「48 核 wall = max-setter（100/100）」。dt 是 12 核 11-worker 量的；
  `_band_env()` 在 12c 與 48c 完全相同（低核分支只在 cores≤8），overlay 一致。
- **48 核形狀的 OOS 是用 `ICCAD_ADAPTIVE_CORES=48` 模擬池組成**，不是真的在 48 核
  機器上跑；真 48 核還會有 wall/RF 效應，本地 eval 一律 RF=1.0。
- 兩個 `_M73_SRC` 候選都是 100 案 in-set 上挑的（組員 7 案、我方 20 案）。§4 說明
  這個自由度幾乎不值錢，但它仍是 in-sample 選擇。
- 12 核的 ΔRF（+3.3~4.5%）是真的。**若評分機有效並行度遠低於 48，這個 tier 由正轉負**
  ——和 tier-5 共用同一個賭注。而且在低核上 tier-5 不觸發，escape 的品質增益會回到
  +0.29% ⇒ 兩者的抵銷關係在低核機器上反轉。這是本結論最大的條件依賴。
- 沒有重推 `_BIG_REDUNDANT_IDX` / `_M45_*`。對 `min_n=100` 的變體**不需要**：
  48 核上 tier-5 已讓 M42 對 n>100 失效、tier-3 只作用於 mid 帶而 tier 不碰 mid，
  兩者不交互；≤16 核的交互則由 in-set 實測涵蓋（5 movers、0 退步）。

## 7. 檔案與復現

| 檔 | 改動 |
|---|---|
| `profile_audit.py` | 第三個 mode `esc`（band overlay，**不套** `_m71_env()`）；`_MODE_KEY` 加 `esc`；esc↔ship 交叉檢查（差異 0 就 abort） |
| `optimizer_constructive.py` | `_M55_IDX` 明確化；`_M73_ESCAPE`(41 隻)/`_M73_BASE`/`_M73_IDX`/`_M73_SRC`/`_m73_active()`；`_pool_indices()` 兩個 tier gate 拆開、都在 `ADAPTIVE_POOL=0` early-return **之前**讀；`ICCAD_M73_MIN_N` 帶別 gate；overlay 抽成 `_profile_env(i, n)`（escape 索引不套 M71 = 機制本體） |
| `m67_oos_probe.py` | `_sig()` 改錨 `_PROFILES[:_M55_BASE_LEN]`（否則 append tier 清空 240 案）；三個 M76 arm；Gate A 的 `elif arm in _M76_ARMS` 分支；**Gate A heavy 期望值改 cores-aware（tier-5）**；`--force-cores`（獨立 cache）；`_csave()` 加 Windows `os.replace` 重試 |
| `m76_escape_probe.py` | **新**，離線工具，`oracle`/`wall`/`derive`/`score`/`report` |

```powershell
python -u profile_audit.py esc > m76_audit_esc.txt          # ~10 分
python -u m76_escape_probe.py oracle
python -u m76_escape_probe.py wall --min-n 100
python -u m76_escape_probe.py derive --cores 48 --band 100,0
python -u m67_oos_probe.py restore --arm m73x --pool0-lo 0                  # 16 核形狀
python -u m67_oos_probe.py run     --force-cores 48                          # 48 形狀基準
python -u m67_oos_probe.py restore --arm m73x --pool0-lo 0 --force-cores 48  # 48 形狀
```

錨檔：`results_M76_gateoff.json`（= M74）、`results_M76_m71off.json`、
`results_M76_tm_allband.json`；OOS dump `results_M72_ab_m73{,big,x}_0_inf.json`。
cache：`audit_cache_esc.pkl`、`m76_cache.pkl`、`m67_oos_cache_c48.pkl`
（16 核形狀的 `m67_oos_cache.pkl` 已備份為 `.preM76`）。

## 8. 對後續的意涵

1. **這條軸關閉**（除非評分機的有效並行度證實遠低於 40，屆時 tier-5 不觸發、
   抵銷關係消失、escape 回到 +0.29%）。
2. **tier-5 比先前認知的更值錢**：它不只買到 M67-F 量到的 +2.289%，還**順帶
   吃掉了整條 escape-tier 軸的 2/3**。
3. **組員這條線可以不用再追**：他們的 +0.288% 在有 tier-5 的包上只剩 ~+0.10%，
   而他們沒有 tier-5 也沒有 audit cache，無法自行發現這件事。
4. 程式碼**保留在 tree 內、預設 off**，當作量測旋鈕（同 M72 tier 的處置）。
