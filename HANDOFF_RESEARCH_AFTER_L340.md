# 研究交接 — L340 結案之後（2026-08-28）

給接手做**研究**的 session。出貨那條線不在這份文件的範圍內，而且**你不應該碰它**。

---

## 0. 先讀這段：出貨已經凍結，研究不可以流進出貨路徑

| | |
|---|---|
| 截止 | **2026-08-31 23:59 GMT+8**（Problem C 專屬延期，見 `DEADLINE_PRIMARY_EVIDENCE.md`） |
| 凍結點 | **08-30 23:59** |
| 已上傳且有效 | `build_submission.D`（rank 2，已驗過，**不做任何事也有分**） |
| 待上傳 | `build_submission.RFSAFE/cadc1075.tar.gz`，`op_wrapper.py` md5 `62db6ee4569b31ddc8c546ccf3e7cd0b`（08-28 複驗過仍一致） |
| 決策 | `SHIP_DECISION_2026-08-28.md` / `UPLOAD_THIS.md` |

**沒有任何研究結果趕得上這次截止。** 任何動到 `constructive.cpp` 的東西都要重建
`bin/constructive_linux`、重跑 Linux 五車道、重走 staging —— 三天內走完的風險遠大於
期望值。而且 `mix` 那條唯一會動名次的臂已經被 L308 量到的 `f = 2.38~2.84` 判掉了
（要 ≥3.0 才贏第一名），剩下的未知數 **截止前結構上不可觀測**。

⇒ **研究一律走 L340 那種隔離方式**：獨立的 `l3xx_*.py` / 獨立的 `.cpp` 與 `.exe`、
不改出貨樹、不重建出貨 ELF、不動 `_PROFILES`。L340 全程如此，出貨樹一個 byte 都沒動。

---

## 1. L340 已結案 —— 品質確立，runtime 判死

完整證據在 `L340_HANDOFF.md`。摘要：

**✅ 品質是真的。** C++ B\*-tree SA（`area + HW·wirelength`，官方評分函式）在 n=80 上
**5/5 個 seed 都贏我們的 packer**，中位數 1.1500 vs 1.2178 = **−0.0678**，最差的 seed
也還有 −0.0508（低於我們 3.9 個 sd）。n=40 / n=120 同方向、同量級。

**🚨 但在預算下三個 n 全輸。** 迭代數 sweep（每點 3 個 seed）：

    在 ~1.4 秒預算內能跑的        打平需要         倍數
    n=40    100k → 1.2104        ~300k / 3.9s     2.8×
    n=80     30k → 1.4425         ~1M / 38.5s    27.5×
    n=120    10k → 1.6512         ~1M / 78.1s    55.8×

預算下的差距是 **+0.096 / +0.225 / +0.438**。**倍數對 n 超線性**（n 變 3 倍、倍數變
20 倍），而 `exp(n/12)` 的權重剛好全壓在最貴那端 ⇒ 兩者往同一個壞方向複合。
C++ 已經比 Python 原型快 100 倍，再要 30–60 倍不是調實作能拿的。
**平行重啟也救不了**（那是 min-of-N，而 n=80/30k 三個 seed 只散在 1.4038~1.4617，
離我們的 1.2178 還差 0.19 ≈ 整個觀測全距的 4 倍；何況我們自己的 1.4 秒本來就在用核）。

**⚠️ 兩個被這輪推翻的說法，不要再引用：**

1. **`HW* = area_L/hpwl_L` 不是最佳點。** gradient 匹配的推導只在**內部**最佳點成立；
   area 已經貼牆（util 0.91~0.93）而 hpwl 沒有 ⇒ 線性配重把**可達前緣**定錯價。
   實測在 n=80 上 `HW*` 是 1.2249，**輸給我們自己的 1.2178**。
2. **「2M 還在爬」在 n=40 上是錯的。** 那是單 seed 量的。3 個 seed 下 n=40 的中位數
   1M→2M 是 **1.0673 → 1.0848（變差）**，早在 1M 就飽和了。還在爬的是 n=80 / n=120。

**🔑 這輪最可外推的一課（請帶著它做任何 sweep）：`min-of-N` 向下偏移**不只套在 seed 上，
**它套在你掃的任何一條軸上**。seed 1 恰好是 HW=2 那格 5 個裡最好的一個，光這件事就
生出了「HW=2 遠優於 HW=4」的排序、n=80 的鋸齒、以及「HW 1→2 兩軸同時變好」的
dominance 假訊號。**改配重只該產生交換，不該產生 dominance —— 兩軸同時變好就是噪音的
長相**，那是當時察覺不對的線索。⇒ **每個 sweep 的 best 都是上界不是估計值**；報數字
一律報等 N 的中位數，並且把格內散布一起報。

---

## 2. 獎品在哪裡（這決定下一條線該打哪）

加權，`mix` 臂：

    hpwl_gap 0.2316  ->  貢獻 0.1158     （area 項的 2.5 倍）
    area_gap 0.0915  ->  貢獻 0.0458
    vrel     0.0139  ->  乘上 1.0283

    只把 area gap 歸零 -> 1.1474      不夠
    只把 HPWL gap 歸零 -> 1.0753      <- 會贏第一名的 1.0845

**獎品在線長，不在面積。** 而 L340 打的是 area+hpwl 的聯合搜尋，L333–L336b 打的是
純面積（生成器自己的目標，`hpwl_gap` 1.13~1.60，方向根本是錯的）。

---

## 3. Tier 1 三條路現在的狀態

原始排序在 `L320_L326_NEW_PATHS.md` §4。三條都動過了，狀態如下：

| | 路徑 | 狀態 | 對 L330 分布位移的曝險 |
|---|---|---|---|
| ① | 重放生成器（B\*-tree SA） | 🚨 **L333–L340 結案**：manifold 是真的、搜尋付不起 | **免疫**（生成器不看 connectivity） |
| ② | netlist 反推當擺放目標 | ⚠️ **L327–L329b：訊號真、管線不成立**（反推中心 + 學到的形狀 hpwl_gap 0.6210，比我們出貨的 0.2402 還差） | **最高** —— 它最依賴 pin 通道，而位移正好全在 pin 通道 |
| ③ | 從 `tree_sol` 監督式預測 B\*-tree | **完全沒動過** | **低**（讀的是版圖結構不是 netlist 密度），但必須查 |

### 🔑 L340 的結果讓 ③ 變成最有意思的一條

L340 證明了**這個 manifold 裡有贏我們的解**（n=80 5/5 seeds），死因**只是搜尋不到**
—— 1.4 秒買不起 1M 次迭代。**監督式預測是「用預測換掉搜尋」**：如果一個模型能直接
跳到一棵好樹，它就繞過了正好殺死 L340 的那道牆。這不是新假說，是 L340 把 ③ 的
價值命題**變乾淨了** —— manifold 的可達性現在是量過的事實，不是猜測。

`tree_sol` 是**訓練 shard 裡直接附的生成器 B\*-tree**（L325 驗證 100%），而比賽的
loader 把它丟掉了。`L320_L326_NEW_PATHS.md` 稱它為「dataset 裡最大的未用資產」。

### 🚨 主辦在 `C_QA_20260827.pdf` A16 直接回答了 ③ 的三個前提（08-28 讀入）

問的人問「訓練集的 `tree_sol` 是不是壞的、會不會出修正版」。主辦的回答有三段對我們有用：

1. **`tree_sol` 是有效的**，但**解碼慣例要對** —— 它與配對的 `fp_sol` 描述同一張版圖，
   前提是用建資料集時那套 tree semantics 與 insertion/traversal order；慣例不同就對不上。
   指定的參考實作是 **IntelLabs/parsac** 的 tree → layout 轉換。
   🔑 **這反過來背書了 L325**：我們的解碼在 `tree_sol` 上驗到 **100%**，代表**我們的慣例
   已經是對的** —— 別隊以為資料壞掉的地方，我們早就解對了。這是 ③ 的前提，現在是確認過的。
2. **「你不需要 `tree_sol` 才能送件」** —— 它是補充資料，所以用它純屬選配，沒有合規疑慮。
3. ⚠️ **主辦明講「For supervised learning or imitation, treat `fp_sol` as the
   ground-truth target layout」** —— 也就是**用 `fp_sol` 當監督是主辦預期的做法**。

**第 3 點與使用者 2026-08-05 的裁示直接相左**（`CLAUDE.md`：完全禁止用 `fp_sol` 當監督，
訓練訊號只能 self-supervised）。那個裁示很可能是在「用 label 當監督不正當」的假設下做的，
而主辦現在白紙黑字說那就是預期做法。**這不是我可以自己翻掉的 —— 但接手的人應該把 A16
拿給使用者看，請他重新裁示一次**，因為整條監督式路線的可用範圍由它決定。

### 🚨 開工前必須先問使用者的兩件事

1. **監督訊號的合法性。** `CLAUDE.md` 記著使用者 2026-08-05 的裁示：
   **「完全禁止用 `fp_sol` 當監督，訓練訊號只能 self-supervised；離線 oracle 探測用
   label 不受限」**。`tree_sol` 是**訓練集（1M shards）**附的，不是 validation label，
   所以**大概**落在允許的一側 —— 但這是使用者的裁示不是我的判斷，**先問再投入**。
2. **preplaced 仍然無解，而且它跟 runtime 無關地擋著送件。** B\*-tree 表達不了固定
   座標。① 死於 runtime，③ **就算預測完美也還是撞這道牆**。所以 ③ 要嘛先解掉
   preplaced（見 `L320_L326_NEW_PATHS.md` Tier 3 ⑦ boundary-constrained B\*-tree、
   ⑧ hierarchical B\*-tree），要嘛就明確定位成「離線上界測量」而不是候選。

### 🚨 開工前先知道這件事（08-28 實測，不要照 `L320_L326` 的字面讀）

**validation 裡沒有 `tree_sol`。** 我今天直接看過檔案結構：

    LiteTensorDataTest/config_80/litelabel_1.pth   2-tuple: metrics (8,) + fp_sol (80,5,2)
    LiteTensorDataTest/config_80/litedata_1.pth    4-tuple: meta / b2b / p2b / pins

沒有 7-tuple、沒有 element [4]。`L320_L326_NEW_PATHS.md` §1.1 寫的是「比賽的 loader
**把樹丟掉**」，那句話容易被讀成「檔案裡有、loader 不給」—— **不是**，1M 訓練 shard
才是 7-tuple，validation 這邊根本沒有那個欄位。

⇒ **監督訊號只在訓練側，評估一定要走「預測 → 解碼 → 官方評分」**，不能用讀樹抄近路。

**所以「先量完美預測值多少」這個便宜的起手式不成立**，而且它本來就是循環的：
validation 有 `fp_sol`，而 L325 證明 label 版圖 100% 服從 B\*-tree 不變量 ⇒ 樹可以從
`fp_sol` 反推 ⇒ 「完美預測那棵樹」就等於「完美重現 label」，那是早就知道的東西
（而且贏過 label 沒分，clamp）。

**非循環的便宜起手式是這個**：L340 的 SA 已經找到**贏我們 packer** 的樹（n=80 5/5
seeds）。量一下**那些好樹離 label 的樹有多遠**。
* 如果近 ⇒ 用 label 樹訓練的模型是瞄準對的地方，③ 的價值命題成立。
* 如果遠 ⇒ **模型會瞄準錯的目標**，因為生成器的目標是純面積，而我們的獎品在線長
  （§2）—— L336b 已經量到重放生成器成品的 `hpwl_gap` 是 **1.13~1.60**，對比我們的 0.240。
  那時整條 ③ 要重新定位成「學表示法、不學目標」。

這個測量不用訓練、用得到的東西 L340 都已經產出了。**先做它再決定要不要投入。**

順帶一個已知的轉移率參考點：L330/L331 量過**形狀**可以從 1M 學到 validation
（held-out 50.8%，比 in-set 掉 3%），所以「訓練集 → validation」這條路本身是通的；
出問題的是 pin 通道（那是 ② 的曝險，不是 ③ 的）。

---

## 4. 環境與陷阱（L340 這輪實際踩到的）

* **`cd ship_final` 每次都要。** `l340_run.py` 有三個相對路徑只在那裡成立：
  `LiteTensorDataTest`（symlink）、`iccad2026contest`、`EXE`。**Bash 的 cwd 會在
  call 之間重置回 `C:\ICCAD_ml`。**
* **`| tee` 會把失敗吞成 exit 0。** 有一輪 sweep 0 秒就死了（`can't open file`）
  卻回報成功。用 `${PIPESTATUS[0]}`。
* python = `C:\Users\.01\anaconda3\envs\floorset\python.exe`。Bash 的 `python` 會
  撞到 Microsoft Store 的假 shim。
* **編譯要 msys 在 PATH，`ICCAD_CXX` 沒用**（指絕對路徑會撞同一個 `cc1plus` 失敗；
  壞的是 `cc1plus` 不是 driver，所以只有 PATH 有效）：

      $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
      g++ -O3 -std=c++17 -o l340_btree.exe l340_btree.cpp

  沒設的話**失敗是無聲的**：整包退回 Python SA，印出 `Total Score: 10.0000` 配
  `Feasible: 100/100`（那個 10.0000 是 SA 的 feasible 上限 9.999999 被 `%.4f` 進位）。
* 量 runtime 時**不要平行跑兩個 sweep** —— 會互相污染（L308 踩過）。

---

## 5. 檔案

    L340_HANDOFF.md            L340 全文（HW sweep / seed / 迭代數 / 兩個更正）
    l340_btree.cpp / .exe      B*-tree + contour decode + SA
    l340_run.py                驅動 + 官方評分       用法見 L340_HANDOFF
    l340_seed.py               等 N 的 seed 散布探針
    l340_iters.py              迭代數 sweep（每點 N 個 seed）

    L320_L326_NEW_PATHS.md     生成器身分、六個不變量、Tier 1/2/3 完整排序、
                               以及「不要花時間的地方」那張明確負面清單
    l320..l336b_*.py           上面那些的探針

⚠️ **`L320_L326_NEW_PATHS.md` §4 末尾那張「Explicit negatives」清單先讀完再提案** ——
AlphaChip、GNN 座標回歸、DREAMPlace、min-cost flow、Lagrangian 形狀規劃、整體 MILP、
multilevel、sequence-pair 對稱 全部已經掃過並判掉，各有理由。

背景記憶：`l340-cpp-btree-sa-inflight`、`l320-l326-generator-identified`、
`l296-l298-gate-depth-compose`（min-of-N）、`aggregate-is-not-its-decomposition`。
