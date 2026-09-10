# M75 — M71 剩下四個 gated-off 旗標：**全部 RED**（2026-07-31）

判準（事前定版）：全域 overlay 形式、**OOS 240 案 gain ≥ +0.3% 且 RF 不惡化**。

錨：OOS 240 `1.576749`（M74 常數）、in-set 100 `1.293461035226291`。

## 結論表

| 旗標 | OOS 240 | in-set 100 | movers | 判定 |
|---|---:|---:|---|---|
| `ICCAD_CLUSTER_BND_CORNER` | **恰 0.0000%** | 恰 0 | 0/240 | **RED（可證明的 no-op）** |
| `ICCAD_ANCHORED_BND_REPACK` | **恰 0.0000%** | 恰 0 | 0/240 | **RED（可證明的 no-op）** |
| `ICCAD_HPWL_SAFE_CLUSTER_SLIDE` | **恰 0.0000%** | 恰 0 | 0/240 | **RED（候選有動、選擇無動）** |
| `ICCAD_CLUSTER_BND_PERMUTE` | **−0.0111%** | +0.0104% | 4/240 | **RED（差 bar 27×，且 OOS 符號相反）** |
| `PERMUTE + SLIDE` | −0.0111% | +0.0104% | 4/240 | 逐位等於 PERMUTE 單獨 |
| 四旗標全聯集 | 同上 | 同上 | 同上 | 矩陣合法收斂，見 §3 |

**送件形零改動。** 這四個旗標維持 default-off，`_M71_ENV` 不變。

## 1. 方法：per-profile liveness map 取代抽樣 arm

原計畫是四個 `m67_oos_probe.py restore` arm，每個 ~15 分鐘。實際做法更嚴格也更快：

先量**每一隻 profile 的 binary 輸出**在旗標 on/off 下是否改變（旗標走 `env_over` 而非
`os.environ`，所以這一步與傳播路徑無關）。若某案的**全部** pool profile 輸出都逐位相同
⇒ 候選集合相同 ⇒ proxy metrics 相同 ⇒ argmin 相同 ⇒ **portfolio 結果逐位相同**。

結果：233/240 OOS 案對全部四個旗標都落在這個「可證明不變」集合裡 ⇒ 整個 delta 只由少數
活案承載 ⇒ 直接解那 21 案（12 OOS + 9 in-set）就得到**精確值**，不是抽樣估計。

| 量測 | 規模 |
|---|---|
| in-set 100 liveness map | 2,460 (case,profile) pairs → 12,300 binary runs，68s |
| OOS 240 liveness map | 5,440 pairs → 27,200 runs，194s |
| 組合確認（baseline = M71+PERMUTE+SLIDE） | 7,900 pairs → 31,600 runs，237s |
| 精確 delta portfolio 解 | 63 solves（21 案 × 3 arm），72s |

合計 **~71,000 次 binary run，0 次失敗**。

## 2. 逐旗標：前件活、效果被吸收

先做了純資料的**前件掃描**（無 exe）。四個前件**全部非空且常見**——所以這**不是** M60 那種
「前件空集」的 RED：

| 旗標 | 前件（OOS 240） | 實際改變輸出的 (case,profile) |
|---|---:|---:|
| CORNER | 67 案 / 79 clusters | **0** |
| REPACK | 216 案 / 402 clusters | **0** |
| PERMUTE | 182 案 / 280 clusters | 27（5 案） |
| SLIDE | 198 案 / 346 clusters | 19（7 案） |

機制解釋（逐行對照 `constructive.cpp`）：

- **CORNER**（`:484-485`）只是往 `orders` 多推一個 `corner_first`。它產生的候選要改變結果，
  必須在 `item_key_better` 的全序下**嚴格勝過** `{boundary_first, by_w, by_h}` 的最佳者。
  340 案 × 全 pool 從未發生一次。
- **REPACK**（`:841-847`）是純評分偏置：`bp==0 && connected` 減 9000、`bp>0` 加 `BP_W`。
  但既有評分已含 `BP_W*bp`（=30000），boundary-clean 候選本來就領先 30000 ⇒ 這個偏置
  **幾乎恆為保序變換**，argmin 不動。這解釋了為何 216/240 有前件卻 0 影響。
- **SLIDE**（`:1483-1565`）的三重 guard（`soft_sig` 不變 ∧ HPWL 嚴格降 ∧ csc 嚴格降）
  加上 `cluster_fits` 的無重疊要求，在 compaction 後的緊密版圖上幾乎必失敗。它確實在
  **52 個 (case,profile)** 上改了輸出，但**全部在小案**（in-set 活案 n=24..53、
  OOS 活案 n ∈ {25,28,33,38,43,54,55}）——而且那些改動的候選**全部輸掉 proxy argmin**
  ⇒ portfolio movers **0/240**。就算它們全贏，`exp(n/12)` 下 n≤55 的權重合計僅約
  總權重的 0.1%，上界仍遠低於 bar。
- **PERMUTE**（`:486-496`）是四者中唯一真正推動分數的。活案 OOS n ∈ {60,73,76,104,115}、
  in-set {54,76,89}。

## 3. 組合矩陣合法收斂（已實測，非僅論證）

使用者要求完整 pairwise + 全聯集。實際只需三個組合，因為：

以 **M71+PERMUTE+SLIDE 為 baseline**，再加 CORNER / REPACK / 兩者，在 in-set 100 +
OOS 240 共 **7,900 個 (case,profile) pair 上改變數 = 0**。

⇒ 四旗標全聯集 **逐位等於** PERMUTE+SLIDE，而 PERMUTE+SLIDE 又逐位等於 PERMUTE。
15 個 arm 的矩陣塌縮成已量的三個。

（論證面也一致：`make_group_item` 的選擇是候選 list 上的嚴格 max，單獨不勝的候選在
任何超集裡仍不勝；REPACK 所在的 anchored first-pass 在 item packing **之前**且不依賴
它。但上面的數字是實測，不是推論。）

## 4. PERMUTE 為什麼是負的——M72 教訓的鏡像

逐案（OOS）：

| n | 案 | shipped → PERMUTE | |
|---:|---|---|---|
| 104 | `worker_2/layouts_4368/L92` | 1.7379 → 1.8174 | **+4.577%（退步）** |
| 76 | `worker_8/layouts_4256/L99` | 1.3469 → 1.3280 | −1.399% |
| 73 | `worker_4/layouts_448/L41` | 1.7689 → 1.4921 | **−15.647%** |
| 60 | `worker_0/layouts_6608/L48` | 1.7157 → 1.7011 | −0.855% |

三個輕案大贏（含一個 −15.6%），但**單一 n=104 重案的 +4.58% 退步在 `exp(n/12)` 下就吃掉全部**。

in-set 側是 **+0.0104%（正的）**、OOS 側 **−0.0111%（負的）**——**符號在兩個語料上就翻了**。
in-set 只有 2 個 mover（case 76 −1.069%、case 54 +1.082%），幾乎互相抵消。

⇒ 若只看 local100 會判成「微弱 GREEN」。這正是 M72 doctrine 的另一面：
**in-sample 不只會藏住 OOS 差距，還會給出相反的符號。**

## 5. 路上修掉／發現的三件事

1. **`m67_oos_probe.py` 的 in-set 錨過期兩代**。`ANCHOR_JSON` 指向
   `results_shipped_m51.json`（檔名誤導：內容是 M71 `1.305389893450635`），
   `IN_SET_TOTAL` 還是 pre-M71 的 `1.326473104916827`，而 tree 從 07-30 起是 M74。
   不修的話 Gate B 會把 **M74 自己的 14 個 movers**（47/49/51/55/62/67/68/77/79/85/87/89/91/96
   ——正是要測的 cluster 硬案）報成 arm 的 movers。已改成
   `results_M74_default.json` / `1.293461035226291`。`_sig()` 不讀這兩個常數 ⇒ cache 中性。

2. **tree 上的 `m67_oos_cache.pkl` 是 M74 某個 sweep 變體的殘留**，sig 對不上現在的碼，
   `gate0` 一載入就清空了 240 案。備份 `m67_oos_cache.pkl.M74k6` 的 sig **完全吻合**
   且有完整 240 案 ⇒ 已還原。教訓：**變體 sweep 跑完要把 live cache 還原成預設組態**，
   否則下一個 session 會靜默付一次全量重跑。

3. **liveness 不能用 portfolio 輸出判**。第一版 Gate A 比最終 positions，四個旗標全報
   「零差異」——但 PERMUTE 明明在 case 89 改了 profile #26、case 76 改了 5 隻。
   旗標可以改變候選卻改不動 proxy 的 argmin。`_m75_liveness()` 已改成 **per-profile** 語意，
   並在 docstring 註明理由。

4. **`_theta_gate_a` 會先把 arm 推進 `os.environ`**，而 `_run_profile` 是
   `env = dict(os.environ); env.update(env_over)` ⇒ 在 Gate A 內做 on/off 比較時，
   **兩邊都是 on**，任何 arm 都會報平的零。修法 = `_m75_liveness()` 進場先 pop、
   `finally` 還原。修好後 Gate A 的數字與獨立跑的 livemap **逐案逐 profile 完全相同**
   （SLIDE case 3 → 23/35p、case 26 → 1/35p；PERMUTE case 54 → 5/20p、76 → 5/20p、
   89 → 1/13p）⇒ 兩套獨立實作互相驗證。CORNER/REPACK 在 Gate A 判 **FAIL 是正確行為**：
   它拒絕為可證明無效的旗標花任何 solve。

## 6. 對 ledger 的意義

- **M71 那條軸到此關閉**。六個旗標裡真正有內容的就是已 ship 的 EXPOSE + EDGE_PACK；
  另外四個在我們的 placer 上是 no-op 或負值。
- 這也解釋了組員 M72 為什麼要把 CORNER/PERMUTE/SLIDE 綁在 `_M55_EXTRA` 的**四隻完整
  profile 配方**裡（連 `FREE_CLUSTER`/`FRAME_SCALES`/`WIRE_MULT` 一起換）——**單獨開這些
  旗標本來就幾乎沒有效果**，他們回報的 +1.287%~+1.531% 是整包配方的增益，不是旗標的。
- 副產物：**M74 的 OOS 240 總分 = `1.576749`**，對 M71 的 `1.586461` 是 **−0.616% OOS**
  （in-set 是 −0.769%）⇒ M74 的增益在 OOS 上同號、幅度相當，不是樣本內過擬合。

## 7. 下一步

四旗標軸已封。正交的下一個候選是**組員的 M73 escape tier**（2026-07-31 fetch 到，
commit `7403758`）：append knob-off 的 profile 副本，讓被 M71 弄壞的案子逃生，
他們量到 local100 `1.301212`（11 好 / 0 壞）、OOS +0.288%。

三點判讀：
1. **我們的 M74 `1.293461` 已經優於他們的 M73 `1.301212`** ⇒ 不是追趕標的。
2. 但機制與 M74 **正交**（M74 動 drop 常數、M73 動池組成）⇒ 值得量。
3. 他們自承 **48 核 wall 未量**，且警告 knob-off 逃生口在大案**可能更慢、可能成為
   max-setter**。他們沒有 `audit_cache.pkl`（handoff 明載「所有 RF 判斷還是推論」），
   **我們有** ⇒ 這題只有我們能結。⚠️ 且 `_M73_SRC=(2,22,23,25)` 是對 M71 的 17 個
   退步案配的，換到 M74 底下必須重推。

## 8. 檔案

- `m67_oos_probe.py` — 錨修正、`_M75_KNOBS`/15 個 arm、Gate A 的 `elif arm in _M75_ARMS`
  分支（pool/band 不變 + `_m71_env()` 完整 + 無 shipped profile 覆蓋 + per-profile liveness）、
  Gate B 的 `expect` 分支。**arm 不進 `_sig()`** ⇒ 既有 cache 全保留。
- `m75_livemap_inset.json` / `m75_livemap_oos.json` — 逐 (case,profile) 活性圖
- `m75_exact.json` — 三個 arm 的精確加權 delta + 逐案 movers
- `m75_livemap_stdout.txt` / `m75_livemap_oos_stdout.txt` / `m75_combo_stdout.txt`
- probe 腳本在 scratchpad（`m75_livemap.py`、`m75_livemap_oos.py`、`m75_precond.py`、
  `m75_exact.py`、`m75_combo_check.py`）——RED 存檔，勿重跑求更好的數字。
